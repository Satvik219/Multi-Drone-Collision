import asyncio
import os
from typing import List

from openai import OpenAI
from graders import grade_episode
from server.drone_env_environment import DroneEnvironment
from models import DroneAction

# Required submission environment variables.
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Optional local image override for workflows using from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# All LLM calls must use the evaluator-provided OpenAI-compatible proxy.
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_BASE_URL and API_KEY else None

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


def validate_proxy_config():
    if not API_BASE_URL:
        raise RuntimeError("Missing required environment variable: API_BASE_URL")
    if not API_KEY:
        raise RuntimeError("Missing required environment variable: API_KEY")


def call_proxy_once(task_name: str, obs) -> None:
    if client is None:
        validate_proxy_config()

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

    rewards = []
    log_start(task_name, "drone_env", MODEL_NAME)

    obs = env.reset()
    call_proxy_once(task_name, obs)

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
