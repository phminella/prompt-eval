import html
import json
import os
import re
import zipfile
from textwrap import dedent
from typing import Any, Literal, TypedDict
from xml.etree import ElementTree


Message = dict[str, str]
JsonDict = dict[str, Any]
Provider = Literal["llama", "openai", "claude", "gemini"]
Mode = Literal["simple", "file"]


class Evaluation(TypedDict):
    score: float
    reasoning: str
    suggestion: str


class PromptEval:
    def __init__(
        self,
        provider: Provider = "llama",
        base_url: str | None = None,
        api_key: str = "not-needed",
        model: str | None = None,
        max_concurrent_tasks: int = 1,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 60,
        messages: list[Message] | None = None,
        stop_sequences: list[str] | None = None,
        system: str | None = None,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_concurrent_tasks = max_concurrent_tasks
        self.messages = messages or []
        self.stop_sequences = stop_sequences or []
        self.system = system
        self.model: str
        self.client: Any

        if provider == "llama":
            from openai import OpenAI

            self.model = model or "local-model"
            self.client = OpenAI(
                base_url=base_url or "http://localhost:8080/v1",
                api_key=api_key,
            )
        elif provider == "openai":
            from openai import OpenAI

            self.model = model or "gpt-4o"
            self.client = OpenAI(
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key,
            )
        elif provider == "claude":
            import anthropic

            self.model = model or "claude-sonnet-4-20250514"
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "gemini":
            from google import genai

            self.model = model or "gemini-2.0-flash"
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Choose from: llama, openai, claude, gemini"
            )

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def chat(
        self,
        messages: list[Message] | None = None,
        system: str | None = None,
        stop_sequences: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        msgs = self.messages if messages is None else messages
        sys_prompt = self.system if system is None else system
        stops = self.stop_sequences if stop_sequences is None else stop_sequences
        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.provider in ("llama", "openai"):
            params: dict[str, Any] = {
                "model": self.model,
                "messages": ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
                + msgs,
                "temperature": temp,
                "max_tokens": tokens,
            }
            if stops:
                params["stop"] = stops

            res = self.client.chat.completions.create(**params)
            return str(res.choices[0].message.content or "")

        if self.provider == "claude":
            claude_params: dict[str, Any] = {
                "model": self.model,
                "messages": msgs,
                "temperature": temp,
                "max_tokens": tokens,
            }
            if sys_prompt:
                claude_params["system"] = sys_prompt
            if stops:
                claude_params["stop_sequences"] = stops

            res = self.client.messages.create(**claude_params)
            return str(res.content[0].text)

        if self.provider == "gemini":
            from google.genai import types

            contents: list[Any] = []
            for msg in msgs:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=msg["content"])])
                )

            config = types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=tokens,
                system_instruction=sys_prompt or None,
                stop_sequences=stops or None,
            )

            res = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return str(res.text or "")

        raise ValueError(f"Unknown provider: {self.provider!r}")

    def _parse_json_response(self, text: str | None) -> JsonDict:
        if text is None:
            raise ValueError("No response from model")

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _judge_result(
        self,
        prompt: str,
        result: str,
        mode: Mode,
        extra_criteria: str | None = None,
    ) -> Evaluation:
        extra_criteria_section = ""
        if extra_criteria:
            extra_criteria_section = f"""
            Extra criteria:
            <extra_criteria>
            {extra_criteria}
            </extra_criteria>
            """

        mode_instruction = {
            "simple": (
                "Evaluate whether the answer correctly satisfies the prompt. "
                "For direct factual or math prompts, an exact correct answer should receive 10."
            ),
            "file": (
                "Evaluate whether the file content is a good result for the prompt that requested it. "
                "Consider completeness, correctness, formatting, and whether the file appears usable."
            ),
        }[mode]

        judge_prompt = f"""
        You are a strict but fair prompt evaluator.

        {mode_instruction}

        Original prompt:
        <prompt>
        {prompt}
        </prompt>

        Result to evaluate:
        <result>
        {result}
        </result>

        {extra_criteria_section}

        Score the result from 1 to 10.

        Also suggest a better version of the original prompt that would improve the answer.
        If the result already deserves a 10, the suggestion can be a slightly clearer equivalent prompt.

        Respond only with valid JSON in this exact shape:
        {{
            "score": number,
            "reasoning": "short explanation",
            "suggestion": "improved prompt"
        }}
        """

        messages = [{"role": "user", "content": dedent(judge_prompt)}]
        text = self.chat(
            messages=messages,
            system="You evaluate prompt results and respond only with valid JSON.",
            temperature=0.0,
        )
        evaluation = self._parse_json_response(text)
        return {
            "score": max(1, min(10, float(evaluation["score"]))),
            "reasoning": str(evaluation["reasoning"]),
            "suggestion": str(evaluation["suggestion"]),
        }

    def _ensure_parent_dir(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _write_json(self, path: str, payload: JsonDict) -> None:
        self._ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _read_result_file(self, result_file: str) -> str:
        extension = os.path.splitext(result_file)[1].lower()

        if extension == ".docx":
            return self._read_docx_text(result_file)

        if extension == ".xlsx":
            return self._read_xlsx_text(result_file)

        with open(result_file, "r", encoding="utf-8") as f:
            return f.read()

    def _read_docx_text(self, result_file: str) -> str:
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with zipfile.ZipFile(result_file) as archive:
            document_xml = archive.read("word/document.xml")

        root = ElementTree.fromstring(document_xml)
        paragraphs = []
        for paragraph in root.findall(".//w:p", namespace):
            text = "".join(
                node.text or "" for node in paragraph.findall(".//w:t", namespace)
            ).strip()
            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    def _read_xlsx_text(self, result_file: str) -> str:
        namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[str] = []
        shared_strings: list[str] = []

        with zipfile.ZipFile(result_file) as archive:
            if "xl/sharedStrings.xml" in archive.namelist():
                shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in shared_root.findall(".//s:si", namespace):
                    shared_strings.append(
                        "".join(
                            node.text or "" for node in item.findall(".//s:t", namespace)
                        )
                    )

            worksheet_names = [
                name
                for name in archive.namelist()
                if name.startswith("xl/worksheets/") and name.endswith(".xml")
            ]
            for worksheet_name in worksheet_names:
                sheet_root = ElementTree.fromstring(archive.read(worksheet_name))
                rows.append(f"Sheet: {os.path.basename(worksheet_name)}")
                for row in sheet_root.findall(".//s:row", namespace):
                    values = []
                    for cell in row.findall("s:c", namespace):
                        cell_type = cell.attrib.get("t")
                        if cell_type == "inlineStr":
                            value = "".join(
                                node.text or ""
                                for node in cell.findall(".//s:t", namespace)
                            )
                        else:
                            value_node = cell.find("s:v", namespace)
                            value = value_node.text if value_node is not None else ""
                            if cell_type == "s" and value:
                                value = shared_strings[int(value)]
                        values.append(value)
                    if values:
                        rows.append(" | ".join(values))

        return "\n".join(rows)

    def generate_file_feedback_report(
        self,
        prompt: str,
        result_file: str,
        file_content: str,
        evaluation: Evaluation,
    ) -> str:
        score = evaluation["score"]
        if score >= 8:
            score_class = "score-high"
        elif score <= 5:
            score_class = "score-low"
        else:
            score_class = "score-medium"

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prompt Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 24px;
                    color: #222;
                    background: #fafafa;
                }}
                main {{
                    max-width: 1100px;
                    margin: 0 auto;
                }}
                section {{
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    margin-bottom: 18px;
                    padding: 18px;
                }}
                h1 {{
                    margin-top: 0;
                }}
                h2 {{
                    margin-top: 0;
                    font-size: 18px;
                }}
                pre {{
                    background: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 12px;
                    overflow: auto;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                .score {{
                    display: inline-block;
                    font-size: 28px;
                    font-weight: bold;
                    padding: 8px 12px;
                    border-radius: 4px;
                }}
                .score-high {{
                    background-color: #c8e6c9;
                    color: #2e7d32;
                }}
                .score-medium {{
                    background-color: #fff9c4;
                    color: #8a6400;
                }}
                .score-low {{
                    background-color: #ffcdd2;
                    color: #c62828;
                }}
            </style>
        </head>
        <body>
            <main>
                <h1>Prompt Evaluation Report</h1>
                <section>
                    <h2>Score</h2>
                    <div class="score {score_class}">{html.escape(str(score))} / 10</div>
                    <p>{html.escape(evaluation["reasoning"])}</p>
                </section>
                <section>
                    <h2>Previous Prompt</h2>
                    <pre>{html.escape(prompt)}</pre>
                </section>
                <section>
                    <h2>New Prompt Suggestion</h2>
                    <pre>{html.escape(evaluation["suggestion"])}</pre>
                </section>
                <section>
                    <h2>Evaluated File</h2>
                    <p>{html.escape(result_file)}</p>
                    <pre>{html.escape(file_content)}</pre>
                </section>
            </main>
        </body>
        </html>
        """

    def run(
        self,
        prompt: str,
        result_prompt: object | None = None,
        result_file: str | None = None,
        extra_criteria: str | None = None,
        json_output_file: str | None = "./eval/output.json",
        html_output_file: str | None = "./eval/output.html",
    ) -> JsonDict:
        """Evaluate either a direct prompt result or the contents of a generated file."""
        if result_prompt is None and result_file is None:
            raise ValueError("Provide either result_prompt or result_file.")

        if result_prompt is not None and result_file is not None:
            raise ValueError("Provide only one of result_prompt or result_file.")

        if result_prompt is not None:
            evaluation = self._judge_result(
                prompt=prompt,
                result=str(result_prompt),
                mode="simple",
                extra_criteria=extra_criteria,
            )
            payload: JsonDict = {
                "mode": "simple",
                "prompt": prompt,
                "result": str(result_prompt),
                **evaluation,
            }

            print("Prompt Evaluation")
            print(f"Score: {evaluation['score']} / 10")
            print(f"Result: {result_prompt}")
            print(f"Reasoning: {evaluation['reasoning']}")
            print(f"Suggested improved prompt: {evaluation['suggestion']}")

            if json_output_file:
                self._write_json(json_output_file, payload)

            return payload

        if result_file is None:
            raise ValueError("Provide result_file for file evaluation.")

        file_content = self._read_result_file(result_file)

        evaluation = self._judge_result(
            prompt=prompt,
            result=file_content,
            mode="file",
            extra_criteria=extra_criteria,
        )
        payload: JsonDict = {
            "mode": "file",
            "prompt": prompt,
            "result_file": result_file,
            "file_content": file_content,
            **evaluation,
        }

        if json_output_file:
            self._write_json(json_output_file, payload)

        if html_output_file:
            self._ensure_parent_dir(html_output_file)
            html_report = self.generate_file_feedback_report(
                prompt=prompt,
                result_file=result_file,
                file_content=file_content,
                evaluation=evaluation,
            )
            with open(html_output_file, "w", encoding="utf-8") as f:
                f.write(html_report)

        print(f"HTML report created: {html_output_file}")
        print(f"Score: {evaluation['score']} / 10")

        return payload
