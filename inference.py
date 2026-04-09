import asyncio
import os
from collections import deque
from typing import List, Tuple

from openai import OpenAI
from graders import grade_episode
from server.drone_env_environment import DroneEnvironment
from models import DroneAction

# ✅ FIXED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

MAX_STEPS = 40
TASKS = ["easy", "medium", "hard"]

Position = Tuple[int, int]


def log_start(task: str):
    print(f"[START] task={task} env=drone_env model={MODEL_NAME}", flush=True)


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


# ---------------- BFS fallback ---------------- #

def _next_direction(start, goal, blocked, grid_size):
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


def _bfs_policy(obs):
    drone_order = sorted(obs.drones.keys())
    all_positions = list(obs.drones.values()) + list(obs.goals.values())
    grid_size = max(max(p) for p in all_positions) + 1
    obstacles = {tuple(o) for o in obs.obstacles}

    for drone in drone_order:
        pos = tuple(obs.drones[drone])
        goal = tuple(obs.goals[drone])
        if pos == goal:
            continue
        occupied = {tuple(p) for d, p in obs.drones.items() if d != drone}
        move = _next_direction(pos, goal, obstacles | occupied, grid_size)
        if move:
            return f"{drone} {move}"

    return f"{drone_order[0]} up"


VALID_MOVES = {"up", "down", "left", "right"}


def _build_prompt(obs):
    drone_lines = "\n".join(
        f"{d}: {tuple(obs.drones[d])} -> {tuple(obs.goals[d])}"
        for d in sorted(obs.drones.keys())
    )
    return f"""Control drones.

{drone_lines}

Return ONLY: <drone_id> <direction>
"""


def llm_policy(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": "Return only: <drone_id> <direction>"},
                {"role": "user", "content": _build_prompt(obs)},
            ],
        )
        raw = response.choices[0].message.content.strip().lower()
        parts = raw.split()

        if len(parts) == 2 and parts[0] in obs.drones and parts[1] in VALID_MOVES:
            return f"{parts[0]} {parts[1]}"

    except Exception as e:
        print(f"[WARN] LLM failed: {e}", flush=True)

    return _bfs_policy(obs)


# ---------------- MAIN ---------------- #

async def run_task(task_name: str):
    env = DroneEnvironment(task_name=task_name)

    rewards = []
    log_start(task_name)

    obs = env.reset()

    path_history = {d: [list(p)] for d, p in obs.drones.items()}
    obstacle_snapshots = [list(obs.obstacles)]

    for step in range(1, MAX_STEPS + 1):
        action = llm_policy(obs)
        result = env.step(DroneAction(command=action))

        obs = result
        rewards.append(result.reward)

        log_step(step, action, result.reward, result.done)

        for d, p in obs.drones.items():
            path_history.setdefault(d, []).append(list(p))
        obstacle_snapshots.append(list(obs.obstacles))

        if result.done:
            break

    grade = grade_episode(
        task_name=task_name,
        final_drones=obs.drones,
        final_goals=obs.goals,
        rewards=rewards,
        steps_taken=len(rewards),
        path_history=path_history,
        obstacle_snapshots=obstacle_snapshots,
    )

    log_end(grade["success"], len(rewards), grade["score"], rewards)


async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())