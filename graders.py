from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

Position = Tuple[int, int]


TASK_CONFIGS = {
    "easy": {
        "starts": {"drone1": (0, 0)},
        "goals": {"drone1": (4, 4)},
        "max_steps": 20,
    },
    "medium": {
        "starts": {"drone1": (0, 0), "drone2": (5, 0)},
        "goals": {"drone1": (5, 5), "drone2": (0, 5)},
        "max_steps": 30,
    },
    "hard": {
        "starts": {"drone1": (0, 0), "drone2": (6, 0)},
        "goals": {"drone1": (6, 6), "drone2": (0, 6)},
        "max_steps": 40,
    },
}


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _total_distance(
    drones: Dict[str, Sequence[int]],
    goals: Dict[str, Sequence[int]],
) -> int:
    return sum(
        _manhattan(
            (int(drones[name][0]), int(drones[name][1])),
            (int(goals[name][0]), int(goals[name][1])),
        )
        for name in drones
    )


def grade_episode(
    task_name: str,
    final_drones: Dict[str, Sequence[int]],
    final_goals: Dict[str, Sequence[int]],
    rewards: Iterable[float],
    steps_taken: int,
) -> Dict[str, float | bool | str]:
    config = TASK_CONFIGS[task_name]
    initial_distance = _total_distance(config["starts"], config["goals"])
    final_distance = _total_distance(final_drones, final_goals)
    drones_total = len(final_drones)
    goals_reached = sum(
        1
        for name, pos in final_drones.items()
        if (int(pos[0]), int(pos[1])) == (int(final_goals[name][0]), int(final_goals[name][1]))
    )

    completion_ratio = goals_reached / drones_total if drones_total else 0.0
    progress_ratio = (
        max(0.0, min(1.0, (initial_distance - final_distance) / initial_distance))
        if initial_distance
        else 1.0
    )
    efficiency_ratio = max(
        0.0,
        min(1.0, 1.0 - (steps_taken / config["max_steps"])),
    )

    score = (
        0.55 * completion_ratio
        + 0.30 * progress_ratio
        + 0.15 * efficiency_ratio
    )
    success = completion_ratio == 1.0
    if success:
        score = max(score, 0.9)

    score = max(0.0, min(1.0, score))

    reward_total = float(sum(rewards))
    return {
        "task": task_name,
        "success": success,
        "score": round(score, 4),
        "completion_ratio": round(completion_ratio, 4),
        "progress_ratio": round(progress_ratio, 4),
        "efficiency_ratio": round(efficiency_ratio, 4),
        "reward_total": round(reward_total, 4),
    }


def grade_easy_episode(
    final_drones: Dict[str, Sequence[int]],
    final_goals: Dict[str, Sequence[int]],
    rewards: List[float],
    steps_taken: int,
) -> Dict[str, float | bool | str]:
    return grade_episode("easy", final_drones, final_goals, rewards, steps_taken)


def grade_medium_episode(
    final_drones: Dict[str, Sequence[int]],
    final_goals: Dict[str, Sequence[int]],
    rewards: List[float],
    steps_taken: int,
) -> Dict[str, float | bool | str]:
    return grade_episode("medium", final_drones, final_goals, rewards, steps_taken)


def grade_hard_episode(
    final_drones: Dict[str, Sequence[int]],
    final_goals: Dict[str, Sequence[int]],
    rewards: List[float],
    steps_taken: int,
) -> Dict[str, float | bool | str]:
    return grade_episode("hard", final_drones, final_goals, rewards, steps_taken)
