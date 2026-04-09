---
title: Multi-Drone Delivery Coordination
emoji: "🚁"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - multi-agent
  - logistics
---

# Multi-Drone Delivery Coordination

This OpenEnv project simulates a real-world drone delivery coordination task. One or more drones must reach delivery targets while respecting grid boundaries, avoiding blocked cells, and not colliding with each other.

## Why this matters

This is not a game-style environment. It is a simplified logistics and fleet-routing problem inspired by warehouse and last-mile delivery systems, where agents must coordinate movement under constraints.

## OpenEnv compliance summary

This repo includes:

- typed `DroneAction` and `DroneObservation` models
- standard `reset()`, `step()`, and `state()` support
- an `openenv.yaml` manifest with three tasks
- explicit Python task graders with normalized scores in `[0.0, 1.0]`
- a deterministic baseline inference script
- a working Dockerfile at `server/Dockerfile`

## Tasks

The environment supports three task levels. Task selection is controlled by `TASK_NAME`.

1. `easy`
- One drone
- No obstacles
- Short horizon
- Goal: reach a single delivery location

2. `medium`
- Two drones
- Light obstacle layout
- Moderate horizon
- Goal: complete both deliveries without collision while balancing fleet workload

3. `hard`
- Two drones
- Denser obstacle layout with a moving blockage
- Longer horizon with tighter routing
- Goal: finish both deliveries efficiently under more constrained and dynamically changing paths

## Agent graders

Task graders are defined in `graders.py`:

- `grade_easy_episode`
- `grade_medium_episode`
- `grade_hard_episode`

Each grader returns:

- `success`: whether the task was completed
- `score`: normalized score in `[0.0, 1.0]`
- `delivery_success`
- `time_efficiency`
- `collision_avoidance`
- `path_optimality`
- `load_balancing`
- `dynamic_obstacles`
- `robust_failure_safe`
- `completion_ratio`
- `progress_ratio`
- `efficiency_ratio`

The final score combines:

- task completion
- time efficiency
- collision avoidance
- route optimality
- fleet load balancing
- dynamic obstacle adaptation
- robust failure-safe execution

This makes the evaluation meaningful even when the agent does not fully solve the task.

## Reward function

The environment uses dense rewards with partial progress signals:

- invalid command or invalid drone id: `-5.0`
- collision attempt: `-10.0`
- obstacle hit: `-5.0`
- valid move: shaped by Manhattan-distance progress toward the acting drone's goal
- reaching a goal: completion bonus
- timeout before completing all deliveries: additional penalty

Because reward is distance-shaped, the agent gets feedback for moving closer to delivery goals instead of only receiving terminal success/failure.

## API surface

This environment exposes the standard OpenEnv interaction pattern:

- `reset()` initializes the selected task and returns the first observation
- `step(action)` applies one action and returns the next observation
- `state()` is available through the OpenEnv server state endpoint and tracks episode metadata such as `episode_id` and `step_count`

## Action space

The action model is `DroneAction` with one typed field:

- `command: str`

Valid actions follow the form:

- `drone1 up`
- `drone1 down`
- `drone1 left`
- `drone1 right`
- `drone2 up`
- `drone2 down`
- `drone2 left`
- `drone2 right`

Only one drone moves per step.

## Observation space

The observation model is `DroneObservation` with these typed fields:

- `drones`: current drone positions keyed by drone id
- `goals`: delivery target positions keyed by drone id
- `obstacles`: blocked cells
- `task_name`: active task id
- `step_count`: number of steps already executed
- `max_steps`: maximum allowed steps for the current task
- `done`: whether the episode has terminated
- `reward`: reward for the latest action

## Key files

- `openenv.yaml`: environment manifest and task metadata
- `models.py`: typed action and observation models
- `graders.py`: task graders and normalized scoring logic
- `server/drone_env_environment.py`: environment logic and reward function
- `inference.py`: deterministic baseline policy and score logging
- `server/app.py`: FastAPI OpenEnv server
- `server/Dockerfile`: Docker build used for deployment

## Run locally

From the environment directory:

```bash
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Task selection:

```bash
TASK_NAME=easy uvicorn server.app:app --host 0.0.0.0 --port 8000
TASK_NAME=medium uvicorn server.app:app --host 0.0.0.0 --port 8000
TASK_NAME=hard uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Baseline inference

Run the deterministic baseline:

```bash
python inference.py
TASK_NAME=easy python inference.py
TASK_NAME=medium python inference.py
TASK_NAME=hard python inference.py
```

Expected log structure:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

The reported `score` is generated by the task graders and is normalized to `[0.0, 1.0]`.

## Validate the environment

```bash
openenv validate
```

## Docker

A working Dockerfile is already included at `server/Dockerfile`.

Build and run locally:

```bash
docker build -t drone-env:latest -f server/Dockerfile .
docker run -p 8000:8000 drone-env:latest
```

## Deploy with OpenEnv

From the environment directory:

```bash
openenv push
```

The deployed Space exposes:

- `/web`
- `/docs`
- `/health`
- `/ws`

## Deploy to Hugging Face Spaces manually

1. Create a new Space on Hugging Face.
2. Select `Docker` as the Space SDK.
3. Give the Space a name and choose public or private visibility.
4. Push this environment directory to the Space repository.
5. Wait for the image build to complete.
6. Open the Space and verify `/health`, `/docs`, and `/web`.

If you need tokens or API keys, add them as Space secrets or variables in the Hugging Face Space settings instead of storing them in the repo.
