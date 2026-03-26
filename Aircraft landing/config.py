from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Aircraft Landing Scheduling Problem.

    Problem Description:
    The problem is to schedule landing times for a set of planes across one or more runways such that each landing occurs within its prescribed time window and all pairwise separation requirements are satisfied; specifically, if plane i lands at or before plane j on the same runway, then the gap between their landing times must be at least the specified separation time provided in the input. In a multiple-runway setting, each plane must also be assigned to one runway, and if planes land on different runways, the separation requirement (which may differ) is applied accordingly. Each plane has an earliest, target, and latest landing time, with penalties incurred proportionally for landing before (earliness) or after (lateness) its target time. The objective is to minimize the total penalty cost while ensuring that no constraints are violated—if any constraint is breached, the solution receives no score.

    Task:
    - Implement the `solve` function that schedules landing times and assigns runways for a set of planes.
    - Each plane has an earliest, target, and latest landing time, with penalties for deviating from the target.
    - If plane i lands at or before plane j on the same runway, the gap between their landing times must be
      at least the specified separation time.
    - The objective is to minimize the total penalty cost while satisfying all constraints.
    - If any constraint is violated, the solution receives no score.

    Input kwargs:
        instance_id : (str) Unique identifier for this problem instance, e.g. "airland1_0".
        num_planes  : (int) Number of planes.
        num_runways : (int) Number of runways.
        freeze_time : (float) Freeze time.
        planes      : (list of dict) Each with keys: "appearance", "earliest", "target", "latest",
                      "penalty_early", "penalty_late".
        separation  : (list of lists) separation[i][j] is the required gap after plane i lands before
                      plane j can land on the same runway.

    Returns:
        A dict with key "schedule" mapping each plane id (1-indexed) to a dict with
        "landing_time" (float) and "runway" (int).

    Key insights to explore:
    1. Greedy scheduling sorted by target or earliest time
    2. Constraint propagation to tighten time windows
    3. Distributing planes across runways to reduce separation conflicts
    4. Local search or metaheuristics (simulated annealing, genetic algorithms)
    5. Mixed-integer programming formulations
    6. Priority-based heuristics using penalty costs to break ties

    IMPORTANT: The main entry point is `def solve(**kwargs)`.
""")


def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]


def build_strategy_model() -> ModelSpec:
    return ModelSpec(
        description="GEMINI 3 Flash Preview",
        model=GoogleModel("gemini-3-flash-preview"),
        settings=GoogleModelSettings(),
    )


def build_evo_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            description="GEMINI 3 Flash Preview",
            model=GoogleModel("gemini-3-flash-preview"),
            settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 8192})
        )
    ]


def get_config(profile: str = "default") -> RayEvolveConfig:
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG,
                                build_strategy_model=build_strategy_model,
                                build_evo_models=build_evo_models, 
                                num_agent_workers=1),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
