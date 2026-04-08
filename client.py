# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Drone Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DroneAction, DroneObservation


class DroneEnv(
    EnvClient[DroneAction, DroneObservation, State]
):
    """Client for the multi-drone delivery coordination environment."""

    def _step_payload(self, action: DroneAction) -> Dict:
        """Convert DroneAction to JSON payload for step message."""
        return {
            "command": action.command,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DroneObservation]:
        """Parse server response into StepResult[DroneObservation]."""
        obs_data = payload.get("observation", {})
        observation = DroneObservation(
            drones=obs_data.get("drones", {}),
            goals=obs_data.get("goals", {}),
            obstacles=obs_data.get("obstacles", []),
            task_name=obs_data.get("task_name", ""),
            step_count=obs_data.get(
                "step_count", payload.get("state", {}).get("step_count", 0)
            ),
            max_steps=obs_data.get("max_steps", 0),
            done=obs_data.get("done", payload.get("done", False)),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
