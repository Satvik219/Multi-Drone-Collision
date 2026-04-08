import asyncio
import os
from typing import List

from openai import OpenAI
from graders import grade_episode
from server.drone_env_environment import DroneEnvironment
from models import DroneAction

# Required submission environment variables.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional local image override for workflows using from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# All LLM calls, if enabled, must use this configured OpenAI-compatible client.
# The baseline policy below remains deterministic and local for reproducibility.
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

MAX_STEPS = 40


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


def simple_policy(obs):
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    drone_order = sorted(obs.drones.keys())

    for drone in drone_order:
        pos = obs.drones[drone]
        goal = obs.goals[drone]

        if tuple(pos) == goal:
            continue

        x, y = pos
        gx, gy = goal

        moves = []
        if x < gx:
            moves.append("down")
        elif x > gx:
            moves.append("up")

        if y < gy:
            moves.append("right")
        elif y > gy:
            moves.append("left")

        occupied_other = [p for d, p in obs.drones.items() if d != drone]

        for move in moves:
            dx, dy = directions[move]
            new_pos = [x + dx, y + dy]

            if new_pos in occupied_other:
                continue

            if hasattr(obs, "obstacles") and tuple(new_pos) in obs.obstacles:
                continue

            return f"{drone} {move}"

    # deterministic fallback for blocked states
    fallback_drone = drone_order[0]
    fx, fy = obs.drones[fallback_drone]
    for move in ("right", "down", "left", "up"):
        dx, dy = directions[move]
        candidate = [fx + dx, fy + dy]
        if hasattr(obs, "obstacles") and tuple(candidate) in obs.obstacles:
            continue
        return f"{fallback_drone} {move}"

    return f"{fallback_drone} up"


async def main():
    task_name = os.getenv("TASK_NAME", "medium").strip().lower() or "medium"
    env = DroneEnvironment(task_name=task_name)

    rewards = []
    log_start(task_name, "drone_env", MODEL_NAME)

    obs = env.reset()

    max_steps = min(MAX_STEPS, getattr(env, "max_episode_steps", MAX_STEPS))

    for step in range(1, max_steps + 1):
        action_str = simple_policy(obs)

        result = env.step(DroneAction(command=action_str))
        obs = result

        reward = result.reward
        done = result.done

        rewards.append(reward)
        log_step(step, action_str, reward, done)

        if done:
            break

    steps_taken = len(rewards)
    grade = grade_episode(
        task_name=task_name,
        final_drones=obs.drones,
        final_goals=obs.goals,
        rewards=rewards,
        steps_taken=steps_taken,
    )
    score = float(grade["score"])
    success = bool(grade["success"])

    log_end(success, step, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
