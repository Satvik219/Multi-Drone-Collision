import os
import json
from typing import List, Tuple

from server.drone_env_environment import DroneEnvironment

MAX_STEPS = 40
Position = Tuple[int, int]


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


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


def run_episode(task: str):
    env = DroneEnvironment(task)
    obs = env.reset()

    rewards = []
    total_reward = 0

    for step in range(MAX_STEPS):
        actions = {}

        for drone, pos in obs.drones.items():
            goal = obs.goals[drone]

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