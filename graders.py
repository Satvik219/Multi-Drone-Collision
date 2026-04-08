from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

Position = Tuple[int, int]


TASK_CONFIGS = {
    "easy": {
        "starts": {"drone1": (0, 0)},
        "goals": {"drone1": (4, 4)},
        "max_steps": 20,
        "grid_size": 5,
        "static_obstacles": [],
    },
    "medium": {
        "starts": {"drone1": (0, 0), "drone2": (5, 0)},
        "goals": {"drone1": (5, 5), "drone2": (0, 5)},
        "max_steps": 30,
        "grid_size": 6,
        "static_obstacles": [(2, 2), (3, 3)],
    },
    "hard": {
        "starts": {"drone1": (0, 0), "drone2": (6, 0)},
        "goals": {"drone1": (6, 6), "drone2": (0, 6)},
        "max_steps": 40,
        "grid_size": 7,
        "static_obstacles": [(2, 2), (3, 3), (4, 4)],
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


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _shortest_path_length(
    start: Position,
    goal: Position,
    grid_size: int,
    obstacles: Sequence[Position],
) -> int:
    if start == goal:
        return 0

    blocked = set(obstacles)
    queue = deque([(start, 0)])
    seen = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            if nxt in blocked or nxt in seen:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))

    return _manhattan(start, goal)


def _count_moves(path: Sequence[Position]) -> int:
    moves = 0
    for idx in range(1, len(path)):
        if path[idx] != path[idx - 1]:
            moves += 1
    return moves


def _incident_count(rewards: Iterable[float], threshold: float) -> int:
    return sum(1 for reward in rewards if reward <= threshold)


def grade_episode(
    task_name: str,
    final_drones: Dict[str, Sequence[int]],
    final_goals: Dict[str, Sequence[int]],
    rewards: Iterable[float],
    steps_taken: int,
    path_history: Dict[str, List[Sequence[int]]] | None = None,
    obstacle_snapshots: List[Sequence[Sequence[int]]] | None = None,
) -> Dict[str, float | bool | str]:
    config = TASK_CONFIGS[task_name]
    reward_list = [float(reward) for reward in rewards]
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
        _clamp_ratio((initial_distance - final_distance) / initial_distance)
        if initial_distance
        else 1.0
    )
    efficiency_ratio = _clamp_ratio(1.0 - (steps_taken / config["max_steps"]))

    optimal_lengths = {
        name: _shortest_path_length(
            config["starts"][name],
            config["goals"][name],
            config["grid_size"],
            config["static_obstacles"],
        )
        for name in config["starts"]
    }
    optimal_total_steps = max(1, sum(optimal_lengths.values()))

    normalized_paths = {
        name: [tuple(map(int, point)) for point in positions]
        for name, positions in (path_history or {}).items()
    }
    actual_moves = {
        name: _count_moves(normalized_paths.get(name, [config["starts"][name], tuple(final_drones[name])]))
        for name in final_drones
    }
    total_actual_moves = max(1, sum(actual_moves.values()))

    delivery_success = completion_ratio
    time_efficiency = efficiency_ratio
    collision_attempts = _incident_count(reward_list, -10.0)
    obstacle_penalties = _incident_count(reward_list, -5.0)
    collision_avoidance = _clamp_ratio(1.0 - (collision_attempts / max(1, steps_taken)))
    path_optimality = _clamp_ratio(optimal_total_steps / total_actual_moves)

    optimal_share = {
        name: optimal_lengths[name] / optimal_total_steps
        for name in optimal_lengths
    }
    actual_share = {
        name: actual_moves[name] / total_actual_moves
        for name in actual_moves
    }
    load_gap = sum(abs(actual_share[name] - optimal_share[name]) for name in actual_share) / 2.0
    load_balancing = _clamp_ratio(1.0 - load_gap)

    obstacle_variations = len(
        {
            tuple(sorted(tuple(map(int, obstacle)) for obstacle in snapshot))
            for snapshot in (obstacle_snapshots or [])
        }
    )
    dynamic_obstacle = 1.0
    if task_name == "hard":
        obstacle_change_ratio = obstacle_variations / max(1, steps_taken + 1)
        dynamic_obstacle = _clamp_ratio(
            0.45 * progress_ratio + 0.35 * collision_avoidance + 0.20 * min(1.0, obstacle_change_ratio * 4.0)
        )

    failure_events = collision_attempts + obstacle_penalties
    robust_failure_safe = _clamp_ratio(1.0 - (failure_events / max(1, steps_taken)))

    score = (
        0.28 * delivery_success
        + 0.16 * time_efficiency
        + 0.14 * collision_avoidance
        + 0.14 * path_optimality
        + 0.10 * load_balancing
        + 0.08 * dynamic_obstacle
        + 0.10 * robust_failure_safe
    )
    success = completion_ratio == 1.0
    if success:
        score = max(score, 0.9)

    score = _clamp_ratio(score)

    reward_total = float(sum(reward_list))
    return {
        "task": task_name,
        "success": success,
        "score": round(score, 4),
        "delivery_success": round(delivery_success, 4),
        "time_efficiency": round(time_efficiency, 4),
        "collision_avoidance": round(collision_avoidance, 4),
        "path_optimality": round(path_optimality, 4),
        "load_balancing": round(load_balancing, 4),
        "dynamic_obstacles": round(dynamic_obstacle, 4),
        "robust_failure_safe": round(robust_failure_safe, 4),
        "completion_ratio": round(completion_ratio, 4),
        "progress_ratio": round(progress_ratio, 4),
        "efficiency_ratio": round(efficiency_ratio, 4),
        "collision_attempts": collision_attempts,
        "obstacle_penalties": obstacle_penalties,
        "optimal_total_steps": optimal_total_steps,
        "actual_total_moves": total_actual_moves,
        "reward_total": round(reward_total, 4),
    }


def grade_easy_episode(
    prediction: Dict[str, object],
    ground_truth: Dict[str, object],
) -> Dict[str, float | bool | str]:
    return grade_episode(
        task_name="easy",
        final_drones=prediction["final_drones"],
        final_goals=ground_truth["final_goals"],
        rewards=prediction["rewards"],
        steps_taken=prediction["steps_taken"],
    )


def grade_medium_episode(
    prediction: Dict[str, object],
    ground_truth: Dict[str, object],
) -> Dict[str, float | bool | str]:
    return grade_episode(
        task_name="medium",
        final_drones=prediction["final_drones"],
        final_goals=ground_truth["final_goals"],
        rewards=prediction["rewards"],
        steps_taken=prediction["steps_taken"],
    )


def grade_hard_episode(
    prediction: Dict[str, object],
    ground_truth: Dict[str, object],
) -> Dict[str, float | bool | str]:
    return grade_episode(
        task_name="hard",
        final_drones=prediction["final_drones"],
        final_goals=ground_truth["final_goals"],
        rewards=prediction["rewards"],
        steps_taken=prediction["steps_taken"],
    )