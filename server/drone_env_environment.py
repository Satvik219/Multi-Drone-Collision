from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ModuleNotFoundError:
    from pydantic import BaseModel

    class Environment:
        """Fallback environment base for local simulation without openenv."""

    class State(BaseModel):
        episode_id: str
        step_count: int = 0

try:
    from ..models import DroneAction, DroneObservation
except ImportError:
    from models import DroneAction, DroneObservation


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class DroneEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "medium"):
        self.task_name = task_name
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.grid_size = 6
        self.max_episode_steps = 30
        self.drones = {}
        self.goals = {}
        self.obstacles = []

    def set_task(self, task_name: str) -> None:
        self.task_name = task_name

    def reset(self) -> DroneObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        if self.task_name == "easy":
            self.grid_size = 5
            self.max_episode_steps = 20
            self.drones = {
                "drone1": [0, 0],
            }
            self.goals = {
                "drone1": (4, 4),
            }
            self.obstacles = []
        elif self.task_name == "medium":
            self.grid_size = 6
            self.max_episode_steps = 30
            self.drones = {
                "drone1": [0, 0],
                "drone2": [5, 0],
            }
            self.goals = {
                "drone1": (5, 5),
                "drone2": (0, 5),
            }
            self.obstacles = [(2, 2), (3, 3)]
        elif self.task_name == "hard":
            self.grid_size = 7
            self.max_episode_steps = 40
            self.drones = {
                "drone1": [0, 0],
                "drone2": [6, 0],
            }
            self.goals = {
                "drone1": (6, 6),
                "drone2": (0, 6),
            }
            self.obstacles = [(2, 2), (2, 3), (3, 3), (4, 4)]
        else:
            # medium is the default task
            self.task_name = "medium"
            self.grid_size = 6
            self.max_episode_steps = 30
            self.drones = {
                "drone1": [0, 0],
                "drone2": [5, 0],
            }
            self.goals = {
                "drone1": (5, 5),
                "drone2": (0, 5),
            }
            self.obstacles = [(2, 2), (3, 3)]

        return DroneObservation(
            drones=self.drones,
            goals=self.goals,
            obstacles=self.obstacles,
            task_name=self.task_name,
            step_count=self._state.step_count,
            max_steps=self.max_episode_steps,
            done=False,
            reward=0.0,
        )

    def step(self, action: DroneAction) -> DroneObservation:
        self._state.step_count += 1

        done = all(tuple(self.drones[d]) == self.goals[d] for d in self.drones)

        try:
            drone, move = action.command.split()
        except Exception:
            return DroneObservation(
                drones=self.drones,
                goals=self.goals,
                obstacles=self.obstacles,
                task_name=self.task_name,
                step_count=self._state.step_count,
                max_steps=self.max_episode_steps,
                done=done,
                reward=-5.0,
            )

        if drone not in self.drones:
            return DroneObservation(
                drones=self.drones,
                goals=self.goals,
                obstacles=self.obstacles,
                task_name=self.task_name,
                step_count=self._state.step_count,
                max_steps=self.max_episode_steps,
                done=done,
                reward=-5.0,
            )

        x, y = self.drones[drone]

        # movement first
        if move == "up":
            x -= 1
        elif move == "down":
            x += 1
        elif move == "left":
            y -= 1
        elif move == "right":
            y += 1
        else:
            return DroneObservation(
                drones=self.drones,
                goals=self.goals,
                obstacles=self.obstacles,
                task_name=self.task_name,
                step_count=self._state.step_count,
                max_steps=self.max_episode_steps,
                done=done,
                reward=-5.0,
            )

        # boundary check
        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))
        new_pos = [x, y]

        # collision detection (-10 reward)
        for other_drone, pos in self.drones.items():
            if other_drone != drone and pos == new_pos:
                return DroneObservation(
                    drones=self.drones,
                    goals=self.goals,
                    obstacles=self.obstacles,
                    task_name=self.task_name,
                    step_count=self._state.step_count,
                    max_steps=self.max_episode_steps,
                    done=done,
                    reward=-10.0,
                )

        # obstacle detection (-5 reward)
        if (x, y) in self.obstacles:
            return DroneObservation(
                drones=self.drones,
                goals=self.goals,
                obstacles=self.obstacles,
                task_name=self.task_name,
                step_count=self._state.step_count,
                max_steps=self.max_episode_steps,
                done=done,
                reward=-5.0,
            )

        previous_distance = _manhattan(tuple(self.drones[drone]), self.goals[drone])

        # apply movement only after passing checks
        self.drones[drone] = new_pos

        # Reward shaping: partial credit for moving closer to the goal.
        new_distance = _manhattan(tuple(new_pos), self.goals[drone])
        progress_delta = previous_distance - new_distance
        reward = -0.2 + (0.8 * progress_delta)

        # Goal completion bonus
        if tuple(new_pos) == self.goals[drone]:
            reward += 5.0

        # done only when all drones reached their goals
        done = all(tuple(self.drones[d]) == self.goals[d] for d in self.drones)

        if not done and self._state.step_count >= self.max_episode_steps:
            done = True
            reward -= 1.0

        return DroneObservation(
            drones=self.drones,
            goals=self.goals,
            obstacles=self.obstacles,
            task_name=self.task_name,
            step_count=self._state.step_count,
            max_steps=self.max_episode_steps,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
