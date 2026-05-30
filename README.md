# prompt-eval

A small Python helper for evaluating prompt outputs with an LLM judge.

It supports two evaluation modes:

- Direct answers, such as checking whether `result_prompt="6"` satisfies `prompt="3+3= ?"`
- Generated files, such as checking whether an HTML, JSON, DOCX, or XLSX file satisfies the prompt that requested it

## Install Locally

From another project, install this package from the local folder:

```powershell
pip install -e *:\****\****
```

## Usage

```python
from prompt_eval import PromptEval

evaluator = PromptEval(provider="llama")

evaluator.run(
    prompt="3+3= ?",
    result_prompt="6",
)
```

For file evaluation:

```python
from prompt_eval import PromptEval

evaluator = PromptEval(provider="llama")

evaluator.run(
    prompt="Create a complete standalone HTML file for a beginner 5K weekly training plan.",
    result_file="data-examples/dataset.html",
)
```

The evaluator writes JSON output to `./eval/output.json`. For file evaluations, it also writes an HTML report to `./eval/output.html`.

## Providers

The default provider is `llama`, using an OpenAI-compatible local server at:

```text
http://localhost:8080/v1
```

Optional providers:

```powershell
pip install -e .[claude]
pip install -e .[gemini]
pip install -e .[all]
```
