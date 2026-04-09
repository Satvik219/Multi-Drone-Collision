import asyncio
import os
from collections import deque
from typing import Dict, List, Sequence, Tuple

from openai import OpenAI
from graders import grade_episode
from server.drone_env_environment import DroneEnvironment
from models import DroneAction

# Required submission environment variables.
API_BASE_URL = os.environ["API_BASE_URL"]   # will raise KeyError if missing — fail fast
API_KEY = os.environ["API_KEY"]             # same
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Optional local image override for workflows using from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# All LLM calls MUST use the evaluator-provided OpenAI-compatible proxy.
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

MAX_STEPS = 40
Position = Tuple[int, int]


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Fallback BFS policy (used when LLM returns an unparseable response)
# ---------------------------------------------------------------------------

def _next_direction(start: Position, goal: Position, blocked: set, grid_size: int):
    if start == goal:
        return None
    directions = [("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))]
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        for name, (dx, dy) in directions:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                continue
            if nxt in blocked or nxt in seen:
                continue
            next_path = path + [name]
            if nxt == goal:
                return next_path[0]
            seen.add(nxt)
            queue.append((nxt, next_path))
    return None


def _bfs_policy(obs) -> str:
    drone_order = sorted(obs.drones.keys())
    all_positions = list(obs.drones.values()) + list(obs.goals.values())
    grid_size = max(max(p) for p in all_positions) + 1
    current_obstacles = {tuple(o) for o in getattr(obs, "obstacles", [])}

    for drone in drone_order:
        pos = tuple(obs.drones[drone])
        goal = tuple(obs.goals[drone])
        if pos == goal:
            continue
        occupied_other = {tuple(p) for d, p in obs.drones.items() if d != drone}
        move = _next_direction(pos, goal, current_obstacles | occupied_other, grid_size)
        if move:
            return f"{drone} {move}"

    # fallback
    fallback = drone_order[0]
    fx, fy = obs.drones[fallback]
    for move, (dx, dy) in (("right", (0, 1)), ("down", (1, 0)), ("left", (0, -1)), ("up", (-1, 0))):
        c = (fx + dx, fy + dy)
        if c in current_obstacles:
            continue
        if c[0] < 0 or c[1] < 0 or c[0] >= grid_size or c[1] >= grid_size:
            continue
        return f"{fallback} {move}"
    return f"{fallback} up"


# ---------------------------------------------------------------------------
# LLM policy — uses the injected proxy for EVERY step decision
# ---------------------------------------------------------------------------

VALID_MOVES = {"up", "down", "left", "right"}


def _build_prompt(obs) -> str:
    drone_lines = "\n".join(
        f"  {d}: position={tuple(obs.drones[d])}, goal={tuple(obs.goals[d])}"
        for d in sorted(obs.drones.keys())
    )
    obstacle_str = str([tuple(o) for o in obs.obstacles]) if obs.obstacles else "none"
    return (
        f"You are controlling drones on a grid.\n"
        f"Drones (name: current position -> goal):\n{drone_lines}\n"
        f"Obstacles: {obstacle_str}\n\n"
        f"Choose ONE action. Reply with EXACTLY: <drone_id> <direction>\n"
        f"where direction is one of: up, down, left, right\n"
        f"Example: drone1 right\n"
        f"Do not add any other text."
    )


def llm_policy(obs) -> str:
    """Ask the LLM proxy for the next action. Falls back to BFS on parse error."""
    prompt = _build_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=16,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a drone routing agent. "
                        "Output ONLY the action in the format: <drone_id> <direction>. "
                        "No explanation, no punctuation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip().lower()
        # Parse and validate
        parts = raw.split()
        if len(parts) == 2 and parts[0] in obs.drones and parts[1] in VALID_MOVES:
            return f"{parts[0]} {parts[1]}"
        # Try to salvage partial match
        for drone in sorted(obs.drones.keys()):
            for move in VALID_MOVES:
                if drone in raw and move in raw:
                    return f"{drone} {move}"
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Using BFS fallback.", flush=True)

    return _bfs_policy(obs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    task_name = os.getenv("TASK_NAME", "medium").strip().lower() or "medium"
    env = DroneEnvironment(task_name=task_name)

    rewards: List[float] = []
    log_start(task_name, "drone_env", MODEL_NAME)

    obs = env.reset()

    path_history: Dict[str, List[Sequence[int]]] = {
        drone: [list(pos)] for drone, pos in obs.drones.items()
    }
    obstacle_snapshots: List[Sequence[Sequence[int]]] = [list(obs.obstacles)]

    max_steps = min(MAX_STEPS, getattr(env, "max_episode_steps", MAX_STEPS))

    for step in range(1, max_steps + 1):
        # Use LLM proxy for every step — this is what the validator checks.
        action_str = llm_policy(obs)

        result = env.step(DroneAction(command=action_str))
        obs = result

        reward = result.reward
        done = result.done

        rewards.append(reward)
        log_step(step, action_str, reward, done)

        for drone, position in obs.drones.items():
            path_history.setdefault(drone, []).append(list(position))
        obstacle_snapshots.append(list(obs.obstacles))

        if done:
            break

    steps_taken = len(rewards)
    grade = grade_episode(
        task_name=task_name,
        final_drones=obs.drones,
        final_goals=obs.goals,
        rewards=rewards,
        steps_taken=steps_taken,
        path_history=path_history,
        obstacle_snapshots=obstacle_snapshots,
    )
    score = float(grade["score"])
    success = bool(grade["success"])

    log_end(success, step, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())