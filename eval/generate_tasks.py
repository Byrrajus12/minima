"""Generate deterministic local evaluation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TASKS_PATH = ROOT / "tasks.json"
EXPECTED_PATH = ROOT / "expected.json"


def task(
    task_id: str,
    category: str,
    prompt: str,
    scoring: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    return (
        {"task_id": task_id, "prompt": prompt},
        {"task_id": task_id, "category": category, "scoring": scoring},
    )


def build_dataset() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows = [
        task(
            "factual_001",
            "factual",
            "What is the capital city of Japan?",
            {"type": "keywords", "keywords": ["tokyo"]},
        ),
        task(
            "factual_002",
            "factual",
            "Which planet is known as the Red Planet?",
            {"type": "keywords", "keywords": ["mars"]},
        ),
        task(
            "factual_003",
            "factual",
            "What gas do plants primarily absorb from the air for photosynthesis?",
            {"type": "keywords", "keywords": ["carbon dioxide", "co2"]},
        ),
        task(
            "factual_004",
            "factual",
            "Who wrote the play Romeo and Juliet?",
            {"type": "keywords", "keywords": ["shakespeare"]},
        ),
        task(
            "factual_005",
            "factual",
            "What is the freezing point of water in Celsius?",
            {"type": "keywords", "keywords": ["0", "zero"]},
        ),
        task(
            "math_001",
            "math",
            "Calculate 18 + 27.",
            {"type": "number", "value": 45, "tolerance": 0},
        ),
        task(
            "math_002",
            "math",
            "What is 9 multiplied by 8?",
            {"type": "number", "value": 72, "tolerance": 0},
        ),
        task(
            "math_003",
            "math",
            "Solve: 144 divided by 12.",
            {"type": "number", "value": 12, "tolerance": 0},
        ),
        task(
            "math_004",
            "math",
            "If a notebook costs 3 dollars, how much do 7 notebooks cost?",
            {"type": "number", "value": 21, "tolerance": 0},
        ),
        task(
            "math_005",
            "math",
            "Calculate 2.5 plus 4.25.",
            {"type": "number", "value": 6.75, "tolerance": 0.001},
        ),
        task(
            "sentiment_001",
            "sentiment",
            "Classify the sentiment: The setup was quick and the result was excellent.",
            {"type": "label", "label": "positive"},
        ),
        task(
            "sentiment_002",
            "sentiment",
            "Classify the sentiment: The tool crashed twice and wasted my time.",
            {"type": "label", "label": "negative"},
        ),
        task(
            "sentiment_003",
            "sentiment",
            "Classify the sentiment: The package arrived at noon.",
            {"type": "label", "label": "neutral"},
        ),
        task(
            "sentiment_004",
            "sentiment",
            "Classify the sentiment: The interface is beautiful, but the export failed.",
            {"type": "label", "label": "mixed"},
        ),
        task(
            "sentiment_005",
            "sentiment",
            "Classify the sentiment: I am disappointed by the slow response.",
            {"type": "label", "label": "negative"},
        ),
        task(
            "summarization_001",
            "summarization",
            "Summarize in one sentence: Lina packed a camera, hiked to the ridge before sunrise, and photographed the valley as fog lifted.",
            {"type": "summary", "keywords": ["lina", "ridge", "fog"], "max_words": 35},
        ),
        task(
            "summarization_002",
            "summarization",
            "Summarize in one sentence: The deployment failed because the service expected a JSON file, but the mounted directory was empty.",
            {"type": "summary", "keywords": ["deployment", "json", "empty"], "max_words": 35},
        ),
        task(
            "summarization_003",
            "summarization",
            "Summarize briefly: Three students compared soil samples, recorded moisture levels, and concluded that shade slowed evaporation.",
            {"type": "summary", "keywords": ["students", "soil", "shade"], "max_words": 35},
        ),
        task(
            "summarization_004",
            "summarization",
            "Summarize briefly: The library extended weekend hours after patrons requested more quiet study time before final exams.",
            {"type": "summary", "keywords": ["library", "weekend", "study"], "max_words": 35},
        ),
        task(
            "summarization_005",
            "summarization",
            "Summarize in one sentence: A small bakery changed suppliers, lowered ingredient costs, and kept its bread prices unchanged.",
            {"type": "summary", "keywords": ["bakery", "suppliers", "prices"], "max_words": 35},
        ),
        task(
            "ner_001",
            "ner",
            "Extract named entities: Maya Patel visited Berlin with Orion Labs in April.",
            {"type": "entities", "entities": ["maya patel", "berlin", "orion labs", "april"]},
        ),
        task(
            "ner_002",
            "ner",
            "Extract named entities: Carlos Rivera joined Northstar Bank in Toronto.",
            {"type": "entities", "entities": ["carlos rivera", "northstar bank", "toronto"]},
        ),
        task(
            "ner_003",
            "ner",
            "Extract named entities: Priya Shah emailed Greenfield School from Mumbai.",
            {"type": "entities", "entities": ["priya shah", "greenfield school", "mumbai"]},
        ),
        task(
            "ner_004",
            "ner",
            "Extract named entities: The River Museum hired Elena Rossi in Florence.",
            {"type": "entities", "entities": ["river museum", "elena rossi", "florence"]},
        ),
        task(
            "ner_005",
            "ner",
            "Extract named entities: Omar Khan presented for Atlas Robotics at Stanford University.",
            {"type": "entities", "entities": ["omar khan", "atlas robotics", "stanford university"]},
        ),
        task(
            "code_debugging_001",
            "code_debugging",
            "Debug this Python code and state the fix: nums=[1,2,3]; print(nums[3])",
            {"type": "substrings", "substrings": ["index", "nums[2]"]},
        ),
        task(
            "code_debugging_002",
            "code_debugging",
            "Debug this Python code and state the fix: def add(a,b): return a-b",
            {"type": "substrings", "substrings": ["+", "return a + b"]},
        ),
        task(
            "code_debugging_003",
            "code_debugging",
            "Debug this Python code and state the fix: for i in range(3) print(i)",
            {"type": "substrings", "substrings": [":", "range(3):"]},
        ),
        task(
            "code_debugging_004",
            "code_debugging",
            "Debug this Python code and state the fix: name='Ada'; print(Name)",
            {"type": "substrings", "substrings": ["case", "name"]},
        ),
        task(
            "code_debugging_005",
            "code_debugging",
            "Debug this Python code and state the fix: value = int('abc')",
            {"type": "substrings", "substrings": ["valueerror", "numeric"]},
        ),
        task(
            "logic_001",
            "logic",
            "If all ravens are birds and Kira is a raven, is Kira a bird?",
            {"type": "label", "label": "yes"},
        ),
        task(
            "logic_002",
            "logic",
            "If no squares are circles and this shape is a square, can it be a circle?",
            {"type": "label", "label": "no"},
        ),
        task(
            "logic_003",
            "logic",
            "Mina is taller than Jo. Jo is taller than Lee. Who is tallest?",
            {"type": "label", "label": "mina"},
        ),
        task(
            "logic_004",
            "logic",
            "A box contains only red balls. Sam picks a ball from the box. What color is it?",
            {"type": "label", "label": "red"},
        ),
        task(
            "logic_005",
            "logic",
            "If the lamp is on, the room is bright. The room is not bright. Is the lamp on?",
            {"type": "label", "label": "no"},
        ),
        task(
            "code_generation_001",
            "code_generation",
            "Write a Python function named double that returns its input multiplied by 2.",
            {"type": "substrings", "substrings": ["def double", "* 2"]},
        ),
        task(
            "code_generation_002",
            "code_generation",
            "Write a Python function named is_even that returns True for even integers.",
            {"type": "substrings", "substrings": ["def is_even", "% 2"]},
        ),
        task(
            "code_generation_003",
            "code_generation",
            "Write a Python function named greet that returns 'Hello, ' plus the provided name.",
            {"type": "substrings", "substrings": ["def greet", "hello"]},
        ),
        task(
            "code_generation_004",
            "code_generation",
            "Write a Python function named first_item that returns the first item in a list.",
            {"type": "substrings", "substrings": ["def first_item", "[0]"]},
        ),
        task(
            "code_generation_005",
            "code_generation",
            "Write a Python function named square that returns n squared.",
            {"type": "substrings", "substrings": ["def square", "n * n"]},
        ),
    ]

    tasks, expected = zip(*rows)
    return list(tasks), list(expected)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def main() -> int:
    tasks, expected = build_dataset()
    write_json(TASKS_PATH, tasks)
    write_json(EXPECTED_PATH, expected)
    print(f"wrote {len(tasks)} tasks to {TASKS_PATH}")
    print(f"wrote scoring metadata to {EXPECTED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
