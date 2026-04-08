import json


# ✅ UNIVERSAL EXTRACTOR (handles platform inconsistencies)
def _extract_prediction_fields(prediction, ground_truth):
    final_drones = prediction.get("final_drones") or prediction.get("drones") or {}

    if isinstance(final_drones, str):
        try:
            final_drones = json.loads(final_drones)
        except:
            final_drones = {}

    final_goals = ground_truth.get("final_goals") or ground_truth.get("goals") or {}

    rewards_raw = prediction.get("rewards", [])
    if isinstance(rewards_raw, str):
        rewards = [float(r) for r in rewards_raw.split(",") if r.strip()]
    else:
        rewards = [float(r) for r in rewards_raw]

    steps_taken = int(
        prediction.get("steps_taken")
        or prediction.get("steps")
        or prediction.get("step_count")
        or 0
    )

    return final_drones, final_goals, rewards, steps_taken


# ✅ MAIN GRADER FUNCTION (USED BY PLATFORM)
def grade_episode(prediction, ground_truth):
    fd, fg, rewards, steps = _extract_prediction_fields(prediction, ground_truth)

    total_distance = 0

    for drone in fd:
        if drone in fg:
            total_distance += abs(fd[drone][0] - fg[drone][0]) + abs(fd[drone][1] - fg[drone][1])

    success = total_distance == 0

    # normalized score
    score = 1.0 if success else max(0, 1 - total_distance / 20)

    return {
        "success": success,
        "score": score,
        "steps": steps,
    }


# ✅ OPTIONAL: wrappers (safe if platform calls specific names)
def grade_easy_episode(prediction, ground_truth):
    return grade_episode(prediction, ground_truth)


def grade_medium_episode(prediction, ground_truth):
    return grade_episode(prediction, ground_truth)


def grade_hard_episode(prediction, ground_truth):
    return grade_episode(prediction, ground_truth)