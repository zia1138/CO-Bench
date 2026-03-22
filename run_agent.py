import enum
from typing import Optional

import typer

from agents import (
    AIDE,
    BestOfN,
    ChainOfExperts,
    DirectAnswer,
    EoHAgent,
    FunSearch,
    GreedyRefine,
    MctsAhd,
    ReEvo,
)
from evaluation import Evaluator, get_data


class AgentChoice(str, enum.Enum):
    AIDE = "AIDE"
    BestOfN = "BestOfN"
    ChainOfExperts = "ChainOfExperts"
    DirectAnswer = "DirectAnswer"
    EoHAgent = "EoHAgent"
    FunSearch = "FunSearch"
    GreedyRefine = "GreedyRefine"
    MctsAhd = "MctsAhd"
    ReEvoAgent = "ReEvoAgent"


app = typer.Typer()


def make_agent(agent: AgentChoice, problem_description: str, timeout: int, model: str):
    common = dict(timeout=timeout, model=model)

    if agent == AgentChoice.AIDE:
        return AIDE(problem_description=problem_description, **common)
    elif agent == AgentChoice.BestOfN:
        return BestOfN(problem_description=problem_description, **common)
    elif agent == AgentChoice.ChainOfExperts:
        return ChainOfExperts(problem_description=problem_description, **common)
    elif agent == AgentChoice.DirectAnswer:
        return DirectAnswer(problem_description=problem_description, **common)
    elif agent == AgentChoice.EoHAgent:
        return EoHAgent(problem=problem_description, llm_model=model)
    elif agent == AgentChoice.FunSearch:
        return FunSearch(problem_description=problem_description, **common)
    elif agent == AgentChoice.GreedyRefine:
        return GreedyRefine(problem_description=problem_description, **common)
    elif agent == AgentChoice.MctsAhd:
        return MctsAhd(problem=problem_description, model=model)
    elif agent == AgentChoice.ReEvoAgent:
        return ReEvo(problem_description=problem_description, **common)


@app.command()
def main(
    dataset: str = typer.Option(..., help="Name of the dataset (e.g. 'Aircraft landing')"),
    agent: AgentChoice = typer.Option(..., help="Agent to use"),
    model: str = typer.Option(
        "google/gemini-3-flash-preview", help="LiteLLM model string"
    ),
    timeout: int = typer.Option(10, help="Timeout in seconds for evaluation"),
    iterations: int = typer.Option(64, help="Maximum number of iterations"),
    src_dir: str = typer.Option("data", help="Directory containing datasets"),
):
    data = get_data(dataset, src_dir=src_dir)

    ag = make_agent(agent, data.problem_description, timeout, model)

    evaluator = Evaluator(data, timeout=timeout)

    for it in range(iterations):
        code = ag.step()
        if code is None:
            break
        feedback = evaluator.evaluate(code)
        print(feedback.dev_feedback)
        ag.feedback(feedback.dev_score, feedback.dev_feedback)

    code = ag.finalize()
    feedback = evaluator.evaluate(code)
    print(feedback.test_feedback)


if __name__ == "__main__":
    app()
