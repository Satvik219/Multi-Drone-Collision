# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Drone Env Environment.

This environment models multi-drone delivery coordination on a grid.
"""

try:
    from openenv.core.env_server.types import Action, Observation
except ModuleNotFoundError:
    from pydantic import BaseModel

    class Action(BaseModel):
        """Fallback Action type for local simulation without openenv."""

    class Observation(BaseModel):
        """Fallback Observation type for local simulation without openenv."""

from pydantic import Field

from typing import Dict, Tuple

class DroneAction(Action):
    command: str = Field(
        ...,
        description="Single command in the form '<drone_id> <direction>', for example 'drone1 up'",
    )

class DroneObservation(Observation):
    drones: Dict[str, list] = Field(..., description="Current drone positions keyed by drone id")
    goals: Dict[str, Tuple[int, int]] = Field(..., description="Target delivery positions for each drone")
    obstacles: list = Field(..., description="Blocked grid cells that drones cannot enter")
    task_name: str = Field(..., description="Current task id: easy, medium, or hard")
    step_count: int = Field(default=0, description="Number of actions executed in the current episode")
    max_steps: int = Field(default=0, description="Maximum allowed steps for the active task")
    done: bool = Field(default=False, description="Whether the current episode has terminated")
    reward: float = Field(default=0.0, description="Dense step reward returned by the environment")
