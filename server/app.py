# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Drone Env Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""
import os

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


class TaskAwareDroneEnvironment(DroneEnvironment):
    """Binds server sessions to the configured task name."""

    def __init__(self):
        task_name = (os.getenv("TASK_NAME", "medium").strip().lower() or "medium")
        if task_name not in {"easy", "medium", "hard"}:
            task_name = "medium"
        super().__init__(task_name=task_name)


app = create_app(
    TaskAwareDroneEnvironment,
    DroneAction,
    DroneObservation,
    env_name="drone_env",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()