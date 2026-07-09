"""Generate a deterministic deeper local evaluation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "eval" / "deep_tasks.json"
EXPECTED_PATH = ROOT / "eval" / "deep_expected.json"


Task = dict[str, str]
Expected = dict[str, Any]


def _item(
    task_id: str,
    category: str,
    prompt: str,
    scoring: dict[str, Any],
) -> tuple[Task, Expected]:
    return (
        {"task_id": task_id, "prompt": prompt},
        {"task_id": task_id, "category": category, "scoring": scoring},
    )


def build_items() -> list[tuple[Task, Expected]]:
    return [
        _item(
            "deep_factual_001",
            "factual",
            "What capital city sits on the Tagus River and is the capital of Portugal?",
            {"type": "keywords", "keywords": ["Lisbon"]},
        ),
        _item(
            "deep_factual_002",
            "factual",
            "What body of water separates Saudi Arabia from northeastern Africa?",
            {"type": "keywords", "keywords": ["Red Sea"]},
        ),
        _item(
            "deep_factual_003",
            "factual",
            "In simple terms, what does a thermostat measure to decide whether heating or cooling is needed?",
            {"type": "keywords", "keywords": ["temperature"]},
        ),
        _item(
            "deep_factual_004",
            "factual",
            "Name the process plants use to convert light, water, and carbon dioxide into sugar.",
            {"type": "keywords", "keywords": ["photosynthesis"]},
        ),
        _item(
            "deep_factual_005",
            "factual",
            "Which planet is known for a prominent ring system and has the moon Titan?",
            {"type": "keywords", "keywords": ["Saturn"]},
        ),
        _item(
            "deep_math_001",
            "math",
            "Calculate: A shop had 48 notebooks, sold 17, then received 3 boxes with 12 notebooks each. How many notebooks are now in stock?",
            {"type": "number", "value": 67, "tolerance": 0},
        ),
        _item(
            "deep_math_002",
            "math",
            "Calculate the sale price: a $80 jacket is discounted by 25%, then a $5 coupon is applied.",
            {"type": "number", "value": 55, "tolerance": 0},
        ),
        _item(
            "deep_math_003",
            "math",
            "Compute the weighted average of scores 90 with weight 2 and 75 with weight 1.",
            {"type": "number", "value": 85, "tolerance": 0},
        ),
        _item(
            "deep_math_004",
            "math",
            "Solve for the projected count: 200 users grow by 10% each month for two months. What is the final user count?",
            {"type": "number", "value": 242, "tolerance": 0},
        ),
        _item(
            "deep_math_005",
            "math",
            "How many minutes are needed for 7 machines to make 84 parts if each machine makes 2 parts per minute?",
            {"type": "number", "value": 6, "tolerance": 0},
        ),
        _item(
            "deep_sentiment_001",
            "sentiment",
            "Classify the sentiment: The camera is sharp and fast, but the battery drains quickly and the case feels cheap.",
            {"type": "label", "label": "mixed"},
        ),
        _item(
            "deep_sentiment_002",
            "sentiment",
            "Classify the sentiment: The installation was effortless, the dashboard is clear, and support answered within minutes.",
            {"type": "label", "label": "positive"},
        ),
        _item(
            "deep_sentiment_003",
            "sentiment",
            "Classify the sentiment: The package arrived broken, the replacement was delayed, and no one replied to my emails.",
            {"type": "label", "label": "negative"},
        ),
        _item(
            "deep_sentiment_004",
            "sentiment",
            "Classify the sentiment: The meeting starts at 2 PM and the agenda has three items.",
            {"type": "label", "label": "neutral"},
        ),
        _item(
            "deep_sentiment_005",
            "sentiment",
            "Classify the sentiment as positive, negative, neutral, or mixed: The soup tasted great, although the service was slow.",
            {"type": "label", "label": "mixed"},
        ),
        _item(
            "deep_summarization_001",
            "summarization",
            "Summarize in one sentence: Riverdale installed flood sensors along the north bank, trained emergency volunteers, and will test sirens next Friday before the rainy season.",
            {"type": "summary", "keywords": ["flood", "sensors", "sirens"], "max_words": 35},
        ),
        _item(
            "deep_summarization_002",
            "summarization",
            "Summarize in exactly two bullets: The library extended evening hours, added a makerspace with 3D printers, and created weekend coding workshops for teens.",
            {"type": "summary", "keywords": ["library", "makerspace", "coding"], "max_words": 55},
        ),
        _item(
            "deep_summarization_003",
            "summarization",
            "Summarize in under 20 words: The bakery now uses solar ovens for morning bread, reducing gas costs while keeping prices unchanged.",
            {"type": "summary", "keywords": ["bakery", "solar", "costs"], "max_words": 20},
        ),
        _item(
            "deep_summarization_004",
            "summarization",
            "Summarize the main takeaway: Metro Health opened a mobile clinic, served 600 patients in two weeks, and plans monthly rural visits.",
            {"type": "summary", "keywords": ["mobile clinic", "600", "rural"], "max_words": 35},
        ),
        _item(
            "deep_summarization_005",
            "summarization",
            "Summarize briefly: The school board approved a garden curriculum after parents donated tools, teachers volunteered training time, and students requested outdoor science labs.",
            {"type": "summary", "keywords": ["garden", "parents", "students"], "max_words": 45},
        ),
        _item(
            "deep_ner_001",
            "ner",
            "Extract named entities: Maya Chen from Orion Foods visited Nairobi on April 3.",
            {"type": "entities", "entities": ["Maya Chen", "Orion Foods", "Nairobi", "April 3"]},
        ),
        _item(
            "deep_ner_002",
            "ner",
            "List the people, organizations, locations, and dates mentioned: Carlos Rivera briefed Northstar Bank in Madrid on June 18.",
            {"type": "entities", "entities": ["Carlos Rivera", "Northstar Bank", "Madrid", "June 18"]},
        ),
        _item(
            "deep_ner_003",
            "ner",
            "Identify named entities: On September 5, Dr. Asha Patel joined GreenGrid Energy for a forum in Vancouver.",
            {"type": "entities", "entities": ["September 5", "Asha Patel", "GreenGrid Energy", "Vancouver"]},
        ),
        _item(
            "deep_ner_004",
            "ner",
            "Find entities: Tomorrow in this report means July 10, 2026, when Lena Ortiz will meet Harbor Analytics in Boston.",
            {"type": "entities", "entities": ["July 10, 2026", "Lena Ortiz", "Harbor Analytics", "Boston"]},
        ),
        _item(
            "deep_ner_005",
            "ner",
            "Extract named entities from this sentence: Quantum Forge hired Daniel Wu in Singapore on January 21.",
            {"type": "entities", "entities": ["Quantum Forge", "Daniel Wu", "Singapore", "January 21"]},
        ),
        _item(
            "deep_code_debugging_001",
            "code_debugging",
            "Debug this Python code and provide the corrected code: def max_value(items):\n    best = items[0]\n    for item in items:\n        return best\nIt should return the largest item.",
            {"type": "substrings", "substrings": ["def max_value", "return best"]},
        ),
        _item(
            "deep_code_debugging_002",
            "code_debugging",
            "Fix the off-by-one bug in this function: def count_down(n):\n    return list(range(n, 0, -1))\nIt should include 0 at the end.",
            {"type": "substrings", "substrings": ["range", "-1"]},
        ),
        _item(
            "deep_code_debugging_003",
            "code_debugging",
            "Correct this Python code that uses a mutable default: def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags",
            {"type": "substrings", "substrings": ["None", "tags = []"]},
        ),
        _item(
            "deep_code_debugging_004",
            "code_debugging",
            "Debug this late-binding Python code: funcs=[]\nfor i in range(3):\n    funcs.append(lambda: i)\nEach lambda should return its own i.",
            {"type": "substrings", "substrings": ["lambda", "i=i"]},
        ),
        _item(
            "deep_code_debugging_005",
            "code_debugging",
            "Fix this generator exhaustion bug: nums = (n for n in [1, 2, 3])\nfirst = sum(nums)\nsecond = sum(nums)\nBoth sums should be 6.",
            {"type": "substrings", "substrings": ["list", "sum"]},
        ),
        _item(
            "deep_logic_001",
            "logic",
            "Logic puzzle: Alice is older than Ben. Ben is older than Cara. Who is the youngest?",
            {"type": "label", "label": "Cara"},
        ),
        _item(
            "deep_logic_002",
            "logic",
            "If all bronze keys open the archive and this key is bronze, does this key open the archive?",
            {"type": "label", "label": "yes"},
        ),
        _item(
            "deep_logic_003",
            "logic",
            "Constraint puzzle: The red box is left of the blue box. The green box is right of the blue box. Which box is in the middle?",
            {"type": "label", "label": "blue"},
        ),
        _item(
            "deep_logic_004",
            "logic",
            "Exactly one of Nora and Omar took the last cookie. Nora says Omar took it. Omar says Nora took it. If Nora tells the truth and Omar lies, who took the cookie?",
            {"type": "label", "label": "Omar"},
        ),
        _item(
            "deep_logic_005",
            "logic",
            "If no electric scooters are allowed in the lobby and Luna's vehicle is an electric scooter, is Luna's vehicle allowed in the lobby?",
            {"type": "label", "label": "no"},
        ),
        _item(
            "deep_code_generation_001",
            "code_generation",
            "Write a Python function named unique_ordered that returns unique items from a list while preserving first-seen order.",
            {"type": "substrings", "substrings": ["def unique_ordered", "seen"]},
        ),
        _item(
            "deep_code_generation_002",
            "code_generation",
            "Implement a Python function named safe_ratio(a, b) that returns None when b is zero, otherwise a / b.",
            {"type": "substrings", "substrings": ["def safe_ratio", "None"]},
        ),
        _item(
            "deep_code_generation_003",
            "code_generation",
            "Create a Python function named flatten_once that flattens one level of nested lists.",
            {"type": "substrings", "substrings": ["def flatten_once", "extend"]},
        ),
        _item(
            "deep_code_generation_004",
            "code_generation",
            "Write a Python function named count_words that returns a dictionary of word counts from a list of strings.",
            {"type": "substrings", "substrings": ["def count_words", "counts"]},
        ),
        _item(
            "deep_code_generation_005",
            "code_generation",
            "Generate Python code for a function named first_or_default(items, default=None) that returns the first item or default for an empty input.",
            {"type": "substrings", "substrings": ["def first_or_default", "default"]},
        ),
    ]


def main() -> int:
    items = build_items()
    tasks = [task for task, _ in items]
    expected = [expected for _, expected in items]
    TASKS_PATH.write_text(json.dumps(tasks, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    EXPECTED_PATH.write_text(
        json.dumps(expected, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} tasks to {TASKS_PATH}")
    print(f"wrote {len(expected)} expected rows to {EXPECTED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
