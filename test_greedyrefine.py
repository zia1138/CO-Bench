from agents import GreedyRefine, DirectAnswer, FunSearch, AIDE, ChainOfExperts, ReEvo, BestOfN
from evaluation import Evaluator, get_data


def main():
    # Load data
    data = get_data('Aircraft landing', src_dir='data')

    # Define agent, here we use GreedyRefine
    agent = GreedyRefine(
        problem_description=data.problem_description,
        timeout=10,
        model='google/gemini-3-flash-preview', # We use LiteLLM to call API
    )

    # Load evaluator
    evaluator = Evaluator(data, timeout=10)

    # Run for 64 iterations
    for it in range(64):
        code = agent.step()
        if code is None:  # agent decides to terminate
            break
        feedback = evaluator.evaluate(code)  # Run evaluation
        agent.feedback(feedback.dev_score, feedback.dev_feedback)  # Use dev set score as feedback

    # Get the final solution
    code = agent.finalize()
    feedback = evaluator.evaluate(code)
    print(feedback.test_feedback)  # Test set score


if __name__ == '__main__':
    main()
