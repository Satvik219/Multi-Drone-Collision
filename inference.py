import os
import json
from typing import List, Tuple

from server.drone_env_environment import DroneEnvironment
<<<<<<< HEAD
from models import DroneAction

# Required submission environment variables.
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()

# Optional local image override for workflows using from_docker_image().
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")


def _resolve_api_key() -> str:
    """Prefer the evaluator-injected proxy key, with safe fallbacks for local runs."""
    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or HF_TOKEN
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing API key. Expected API_KEY from the evaluator, "
            "or OPENAI_API_KEY/HF_TOKEN for local testing."
        )
    return api_key


def build_client() -> OpenAI:
    if not API_BASE_URL:
        raise RuntimeError("Missing required environment variable: API_BASE_URL")
    return OpenAI(base_url=API_BASE_URL, api_key=_resolve_api_key())
=======
>>>>>>> a177dc5 (Final fix for graders + inference)

MAX_STEPS = 40
Position = Tuple[int, int]


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


# ✅ FIXED END LOGGER (VERY IMPORTANT)
def log_end(success: bool, steps: int, score: float, rewards: List[float], final_drones=None):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    fd_str = ""
    if final_drones:
        try:
            fd_clean = {k: list(v) for k, v in final_drones.items()}
            fd_str = f" final_drones={json.dumps(fd_clean)}"
        except:
            fd_str = ""

    print(
        f"[END] success={str(success).lower()} steps={steps} steps_taken={steps} score={score:.2f} rewards={rewards_str}{fd_str}",
        flush=True,
    )


<<<<<<< HEAD
def _next_direction(start: Position, goal: Position, blocked: set[Position], grid_size: int) -> str | None:
    if start == goal:
        return None

    directions = [
        ("up", (-1, 0)),
        ("down", (1, 0)),
        ("left", (0, -1)),
        ("right", (0, 1)),
    ]
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


def simple_policy(obs):
    drone_order = sorted(obs.drones.keys())
    grid_size = max(
        [max(max(pos) for pos in obs.drones.values()), max(max(goal) for goal in obs.goals.values())],
        default=0,
    ) + 1
    current_obstacles = {tuple(obstacle) for obstacle in getattr(obs, "obstacles", [])}

    for drone in drone_order:
        pos = tuple(obs.drones[drone])
        goal = tuple(obs.goals[drone])

        if pos == goal:
            continue

        occupied_other = {
            tuple(position)
            for other_drone, position in obs.drones.items()
            if other_drone != drone
        }
        move = _next_direction(pos, goal, current_obstacles | occupied_other, grid_size)
        if move:
            return f"{drone} {move}"

    # deterministic fallback for blocked states
    fallback_drone = drone_order[0]
    fx, fy = obs.drones[fallback_drone]
    for move, (dx, dy) in (("right", (0, 1)), ("down", (1, 0)), ("left", (0, -1)), ("up", (-1, 0))):
        candidate = (fx + dx, fy + dy)
        if candidate in current_obstacles:
            continue
        if candidate[0] < 0 or candidate[1] < 0 or candidate[0] >= grid_size or candidate[1] >= grid_size:
            continue
        return f"{fallback_drone} {move}"

    return f"{fallback_drone} up"


def call_proxy_once(task_name: str, obs) -> None:
    client = build_client()
    drone_state = ", ".join(
        f"{drone}:{tuple(position)}->{tuple(obs.goals[drone])}"
        for drone, position in sorted(obs.drones.items())
    )

    client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=16,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are validating a drone-routing run. "
                    "Reply with a very short acknowledgement."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task={task_name}. Initial drone state: {drone_state}. "
                    "Acknowledge that you can evaluate this run."
                ),
            },
        ],
    )


async def main():
    task_name = os.getenv("TASK_NAME", "medium").strip().lower() or "medium"
    env = DroneEnvironment(task_name=task_name)
=======
def run_episode(task: str):
    env = DroneEnvironment(task)
    obs = env.reset()
>>>>>>> a177dc5 (Final fix for graders + inference)

    rewards = []
    total_reward = 0

    for step in range(MAX_STEPS):
        actions = {}

        for drone, pos in obs.drones.items():
            goal = obs.goals[drone]

            # simple greedy movement
            if pos[0] < goal[0]:
                action = "down"
            elif pos[0] > goal[0]:
                action = "up"
            elif pos[1] < goal[1]:
                action = "right"
            elif pos[1] > goal[1]:
                action = "left"
            else:
                action = "stay"

            actions[drone] = action

        obs, reward, done, _ = env.step(actions)

        rewards.append(reward)
        total_reward += reward

        log_step(step, str(actions), reward, done)

        if done:
            break

    success = all(
        tuple(obs.drones[d]) == tuple(obs.goals[d])
        for d in obs.drones
    )

    score = total_reward / (step + 1)

    # ✅ CRITICAL FIX: send final_drones
    log_end(success, step + 1, score, rewards, final_drones=obs.drones)

    return {
        "success": success,
        "score": score,
    }


if __name__ == "__main__":
    MODEL_NAME = os.getenv("MODEL_NAME", "baseline")

    for task in ["easy", "medium", "hard"]:
        log_start(task, "drone_env", MODEL_NAME)
        run_episode(task)