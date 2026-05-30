from prompt_eval import PromptEval

evaluator = PromptEval()
results = evaluator.run(
    prompt="Create a HTML file containing a Beginner 5K 3-day a week Training Plan",
    result_file="data-examples/dataset.html",
)
