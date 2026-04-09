# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Drone Env Environment.

Endpoints:
    - POST /reset: Reset the environment (accepts optional task_name in body)
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /tasks: List all available tasks
    - POST /grader: Grade an episode result
    - WS /ws: WebSocket endpoint for persistent sessions
"""
import os
from typing import Any, Dict

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import DroneAction, DroneObservation
    from .drone_env_environment import DroneEnvironment
except ModuleNotFoundError:
    from models import DroneAction, DroneObservation
    from server.drone_env_environment import DroneEnvironment

try:
    from graders import (
        TASK_CONFIGS,
        grade_easy_episode,
        grade_medium_episode,
        grade_hard_episode,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graders import (
        TASK_CONFIGS,
        grade_easy_episode,
        grade_medium_episode,
        grade_hard_episode,
    )

from pydantic import BaseModel

_GRADER_MAP = {
    "easy": grade_easy_episode,
    "medium": grade_medium_episode,
    "hard": grade_hard_episode,
}

_GROUND_TRUTH = {
    "easy":   {"final_goals": {"drone1": [4, 4]}},
    "medium": {"final_goals": {"drone1": [5, 5], "drone2": [0, 5]}},
    "hard":   {"final_goals": {"drone1": [6, 6], "drone2": [0, 6]}},
}


class TaskAwareDroneEnvironment(DroneEnvironment):
    """Binds server sessions to the configured task name, with per-reset override."""

    def __init__(self):
        task_name = (os.getenv("TASK_NAME", "medium").strip().lower() or "medium")
        if task_name not in {"easy", "medium", "hard"}:
            task_name = "medium"
        super().__init__(task_name=task_name)


# Create the base app with web interface
app = create_app(
    TaskAwareDroneEnvironment,
    DroneAction,
    DroneObservation,
    env_name="drone_env",
    max_concurrent_envs=10,
)


# ── /tasks endpoint ─────────────────────────────────────────────────────────

@app.get("/tasks", tags=["Tasks"], summary="List available tasks")
def list_tasks() -> Dict[str, Any]:
    """Return all task definitions with their grader metadata."""
    tasks = []
    for task_id, cfg in TASK_CONFIGS.items():
        tasks.append({
            "id": task_id,
            "difficulty": task_id,
            "max_steps": cfg["max_steps"],
            "grid_size": cfg["grid_size"],
            "num_drones": len(cfg["starts"]),
            "ground_truth": _GROUND_TRUTH[task_id],
            "grader": {
                "type": "python",
                "entrypoint": f"graders:grade_{task_id}_episode",
                "score_range": [0.0, 1.0],
                "score_key": "score",
            },
        })
    return {"tasks": tasks}


# ── /grader endpoint ─────────────────────────────────────────────────────────

class GraderRequest(BaseModel):
    task_id: str
    prediction: Dict[str, Any]
    ground_truth: Dict[str, Any] = {}


@app.post("/grader", tags=["Tasks"], summary="Grade an episode result")
def grade_episode_endpoint(request: GraderRequest) -> Dict[str, Any]:
    """
    Grade a prediction against ground truth for a given task.
    Accepts prediction dict with keys: final_drones, rewards, steps_taken.
    """
    task_id = request.task_id.strip().lower()
    if task_id not in _GRADER_MAP:
        return {"error": f"Unknown task_id '{task_id}'. Valid: easy, medium, hard"}

    grader_fn = _GRADER_MAP[task_id]
    ground_truth = request.ground_truth or _GROUND_TRUTH[task_id]

    try:
        result = grader_fn(prediction=request.prediction, ground_truth=ground_truth)
        return dict(result)
    except Exception as exc:
        return {"error": str(exc)}


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()