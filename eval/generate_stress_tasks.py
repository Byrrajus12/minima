"""Generate a deterministic 160-task stress evaluation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "eval" / "stress_tasks.json"
EXPECTED_PATH = ROOT / "eval" / "stress_expected.json"

DIFFICULTIES = ("easy", "medium", "hard", "adversarial")


def _difficulty(index: int) -> str:
    return DIFFICULTIES[(index - 1) % len(DIFFICULTIES)]


def _add(
    rows: list[tuple[dict[str, str], dict[str, Any]]],
    category: str,
    index: int,
    prompt: str,
    check: dict[str, Any],
    notes: str,
) -> None:
    task_id = f"stress_{category}_{index:03d}"
    rows.append(
        (
            {"task_id": task_id, "prompt": prompt},
            {
                "task_id": task_id,
                "category": category,
                "difficulty": _difficulty(index),
                "check": check,
                "notes": notes,
            },
        )
    )


def _factual(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    items = [
        ("What is the capital of Chile?", ["Santiago"], "single common-knowledge fact"),
        ("What natural process lets plants turn light, water, and carbon dioxide into sugar?", ["photosynthesis"], "definition"),
        ("Name the capital of Egypt and the river that runs through it.", ["Cairo", "Nile"], "two required details"),
        ("In simple terms, what does an odometer measure in a car?", ["distance"], "how a familiar device works"),
        ("What body of water lies between Florida and Mexico?", ["Gulf of Mexico"], "geography body of water"),
        ("What gas do humans mainly exhale after cells use oxygen?", ["carbon dioxide"], "basic biology"),
        ("Which city is the capital of Japan, and which island is it on?", ["Tokyo", "Honshu"], "two-detail geography"),
        ("What does a thermometer measure?", ["temperature"], "definition"),
        ("Which planet has the Great Red Spot?", ["Jupiter"], "planet fact"),
        ("What force pulls objects toward Earth?", ["gravity"], "physics definition"),
        ("Name the capital of Italy and the country completely surrounded by that city.", ["Rome", "Vatican"], "two related facts"),
        ("What do bees collect from flowers to make honey?", ["nectar"], "natural process"),
        ("Which ocean borders California?", ["Pacific"], "geography"),
        ("What is evaporation?", ["liquid", "gas"], "concise explanation"),
        ("Who wrote the play Hamlet?", ["Shakespeare"], "literary fact"),
        ("What organ pumps blood through the human body?", ["heart"], "body system"),
        ("Name the capital of Kenya and the country it belongs to.", ["Nairobi", "Kenya"], "two details"),
        ("How does a compass help with navigation?", ["magnetic", "north"], "how things work"),
        ("What is the freezing point of water in Celsius?", ["0"], "numeric factual answer"),
        ("Which country is both a continent and an island nation?", ["Australia"], "common knowledge"),
    ]
    for index, (prompt, keywords, notes) in enumerate(items, start=1):
        _add(rows, "factual", index, prompt, {"type": "contains_all", "values": keywords}, notes)


def _math(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    items = [
        ("Calculate: 18 + 27 - 9.", 36, "arithmetic"),
        ("A store had 120 pens, sold 45, then received 30 more. How many pens are there now?", 105, "inventory remaining"),
        ("Calculate the sale price: a $50 item is discounted by 20%.", 40, "percentage discount"),
        ("Compute the weighted average of 80 with weight 3 and 95 with weight 1.", 83.75, "weighted average"),
        ("Calculate the final population: 500 grows by 10% for two years.", 605, "projection"),
        ("If 4 workers make 48 parts in 6 hours, how many parts per worker per hour?", 2, "rate problem"),
        ("Calculate the total cost of 3 notebooks at $4 each and 2 pens at $1.50 each.", 15, "multi-item cost"),
        ("A recipe uses 2 cups of flour for 8 muffins. How many cups are needed for 20 muffins?", 5, "proportion"),
        ("Solve: 7 * 8 - 13.", 43, "arithmetic order"),
        ("A class has 12 girls and 18 boys. What percentage are girls?", 40, "percentage"),
        ("Calculate the speed in miles per hour: a runner covers 15 miles in 3 hours.", 5, "rate"),
        ("A box has 9 rows of 6 tiles, and 8 tiles break. How many usable tiles remain?", 46, "inventory with loss"),
        ("An account has $200 and earns 5% interest once. What is the balance?", 210, "interest"),
        ("The average of 10, 14, and x is 15. Solve for x.", 21, "solve for variable"),
        ("A train travels 180 miles at 60 mph. How many hours does it take?", 3, "time rate distance"),
        ("A jacket costs $90 after a 10% discount. What was the original price?", 100, "reverse percentage"),
        ("Compute 30% of 250.", 75, "percentage of quantity"),
        ("Calculate the fraction of games won: a team won 14 games and lost 6.", 0.7, "fraction rate"),
        ("There are 5 packs with 12 batteries each. If 17 are used, how many remain?", 43, "multi-step count"),
        ("A price rises from 80 to 100. What is the percent increase?", 25, "percent increase"),
    ]
    for index, (prompt, value, notes) in enumerate(items, start=1):
        _add(rows, "math", index, prompt, {"type": "numeric_exact", "value": value, "tolerance": 0.01}, notes)


def _sentiment(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    items = [
        ("Classify the sentiment: The meal was delicious and the staff were friendly.", "positive", False, "clear positive"),
        ("Classify the sentiment: The app crashed twice and support never replied.", "negative", False, "clear negative"),
        ("Classify the sentiment: The report lists quarterly revenue and operating expenses.", "neutral", False, "neutral statement"),
        ("Classify the sentiment: The headphones sound excellent, but the headband broke after two days.", "mixed", False, "substantial positive and negative"),
        ("Classify the sentiment and briefly justify: The room was spotless and quiet, though the Wi-Fi barely worked.", "mixed", True, "mixed with justification requested"),
        ("Classify the sentiment: I love the screen, speed, and battery life.", "positive", False, "positive short review"),
        ("Classify the sentiment: The delivery was late, the box was wet, and the product was scratched.", "negative", False, "negative multiple issues"),
        ("Classify the sentiment: The meeting is scheduled for Monday at noon.", "neutral", False, "neutral schedule"),
        ("Classify as positive, negative, neutral, or mixed: Great camera quality, awful low-light autofocus.", "mixed", False, "compact mixed"),
        ("Classify the sentiment and give one reason: The shoes look stylish and fit well, but the sole separated immediately.", "mixed", True, "mixed with explicit reason"),
        ("Classify the sentiment: The update made search faster and fixed the login issue.", "positive", False, "product improvement"),
        ("Classify the sentiment: The new policy is described in section four.", "neutral", False, "descriptive neutral"),
        ("Classify the sentiment: Every page loaded slowly and the checkout failed.", "negative", False, "negative usability"),
        ("Classify the sentiment: The concert started late, but the performance itself was outstanding.", "mixed", False, "mixed event review"),
        ("Classify the sentiment and justify briefly: The software is powerful, yet the onboarding is confusing.", "mixed", True, "mixed software review"),
        ("Classify the sentiment: This chair is comfortable, sturdy, and easy to assemble.", "positive", False, "positive product"),
        ("Classify the sentiment: The warranty terms changed on July 1.", "neutral", False, "neutral policy"),
        ("Classify the sentiment: The soup was cold and the server ignored us.", "negative", False, "negative dining"),
        ("Classify as positive, negative, neutral, or mixed: Helpful documentation, but the examples are outdated.", "mixed", False, "mixed documentation"),
        ("Classify the sentiment and include a short justification: Beautiful design, unreliable charging.", "mixed", True, "short mixed with justification"),
    ]
    for index, (prompt, label, require_reason, notes) in enumerate(items, start=1):
        _add(
            rows,
            "sentiment",
            index,
            prompt,
            {"type": "label_with_optional_reason", "label": label, "require_reason": require_reason},
            notes,
        )


def _summarization(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    items = [
        ("Summarize in one sentence: The city repaired the bridge, reopened two bus routes, and plans overnight inspections next month.", ["bridge", "bus", "inspections"], {"sentence_count": 1, "max_words": 35}, "one sentence plus facts"),
        ("Summarize in exactly two bullets: The museum added Friday evening hours, opened a textile exhibit, and launched free student tours.", ["museum", "textile", "student"], {"bullet_count": 2, "max_words": 60}, "two bullets"),
        ("Summarize in under 18 words: A bakery switched to electric ovens, lowered fuel costs, and kept bread prices unchanged.", ["bakery", "ovens", "costs"], {"max_words": 18}, "word cap"),
        ("Summarize briefly: The clinic vaccinated 400 residents, added weekend appointments, and hired two nurses.", ["clinic", "400", "nurses"], {"max_words": 35}, "preserve numbers"),
        ("Summarize in one sentence: The school garden program expanded after parents donated tools and students requested outdoor science labs.", ["garden", "parents", "students"], {"sentence_count": 1, "max_words": 35}, "community details"),
        ("Summarize in exactly two bullets: The airport upgraded scanners, reduced average security waits, and added multilingual signs.", ["airport", "security", "signs"], {"bullet_count": 2, "max_words": 55}, "format"),
        ("Summarize in fewer than 20 words: The team delayed the launch to fix payment bugs and improve account recovery.", ["launch", "payment", "recovery"], {"max_words": 20}, "compact product summary"),
        ("Summarize in one sentence: Farmers tested drought-resistant corn, recorded higher yields, and shared seed data with nearby villages.", ["corn", "yields", "villages"], {"sentence_count": 1, "max_words": 35}, "agriculture facts"),
        ("Summarize briefly without adding unsupported details: The council approved bike lanes on Pine Street and postponed the parking vote.", ["bike", "Pine", "parking"], {"max_words": 35}, "avoid hallucination"),
        ("Summarize in exactly two bullets: The app added offline maps, compressed downloads, and a route-sharing feature.", ["offline", "downloads", "route"], {"bullet_count": 2, "max_words": 55}, "two bullets product"),
        ("Summarize in one sentence: A storm damaged the pier, but crews restored power and reopened the market by Sunday.", ["storm", "power", "market"], {"sentence_count": 1, "max_words": 35}, "contrast"),
        ("Summarize under 25 words: The university froze tuition, expanded emergency grants, and created a food pantry partnership.", ["tuition", "grants", "pantry"], {"max_words": 25}, "three facts"),
        ("Summarize in one sentence: The theater sold out opening night, added two performances, and donated proceeds to flood relief.", ["theater", "performances", "flood"], {"sentence_count": 1, "max_words": 35}, "event summary"),
        ("Summarize in exactly two bullets: The zoo opened a night exhibit, trained new guides, and capped visitor numbers.", ["zoo", "guides", "visitor"], {"bullet_count": 2, "max_words": 55}, "two bullets"),
        ("Summarize briefly: The company recalled the charger after overheating reports and offered free replacements through August.", ["charger", "overheating", "August"], {"max_words": 35}, "recall details"),
        ("Summarize in one sentence: The library digitized local newspapers, added searchable archives, and invited residents to tag photos.", ["library", "archives", "photos"], {"sentence_count": 1, "max_words": 35}, "archive facts"),
        ("Summarize under 20 words: The cafe introduced reusable cups and gives discounts to customers who bring them back.", ["cafe", "reusable", "discounts"], {"max_words": 20}, "environmental action"),
        ("Summarize in exactly two bullets: The hospital opened a pediatric wing, hired specialists, and shortened referral times.", ["hospital", "pediatric", "referral"], {"bullet_count": 2, "max_words": 55}, "healthcare"),
        ("Summarize in one sentence: The committee rejected the budget draft because travel costs rose and revenue estimates fell.", ["budget", "travel", "revenue"], {"sentence_count": 1, "max_words": 35}, "cause"),
        ("Summarize briefly: The rail agency tested quieter brakes, surveyed nearby residents, and will publish findings in October.", ["brakes", "residents", "October"], {"max_words": 35}, "transport facts"),
    ]
    for index, (prompt, keywords, constraints, notes) in enumerate(items, start=1):
        check = {"type": "summary_constraints", "keywords": keywords}
        check.update(constraints)
        _add(rows, "summarization", index, prompt, check, notes)


def _ner(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    names = [
        ("Maya Chen", "Orion Foods", "Nairobi", "April 3"),
        ("Carlos Rivera", "Northstar Bank", "Madrid", "June 18"),
        ("Asha Patel", "GreenGrid Energy", "Vancouver", "September 5"),
        ("Lena Ortiz", "Harbor Analytics", "Boston", "July 10, 2026"),
        ("Daniel Wu", "Quantum Forge", "Singapore", "January 21"),
        ("Priya Shah", "Atlas Robotics", "Toronto", "July 9"),
        ("Elena Morris", "BrightWave Labs", "Lisbon", "March 12"),
        ("Noah Bennett", "Cedar Health", "Denver", "May 4"),
        ("Iris Kim", "Blue Harbor Studio", "Seoul", "August 14"),
        ("Omar Ali", "Summit Rail", "Cairo", "February 2"),
        ("Grace Lin", "Nova Textiles", "Osaka", "November 6"),
        ("Ethan Brooks", "Maple Trust", "Chicago", "October 19"),
        ("Sofia Rossi", "Argo Marine", "Athens", "December 8"),
        ("Mateo Cruz", "Silverline Media", "Bogota", "March 30"),
        ("Hannah Park", "Zenith Solar", "Phoenix", "June 1"),
        ("Nina Kapoor", "Riverstone Analytics", "Dublin", "April 22"),
        ("Theo Martin", "Pioneer Foods", "Paris", "January 5"),
        ("Amara Okafor", "Lumen Bank", "Lagos", "May 17"),
        ("Jonas Meyer", "Northwind Games", "Berlin", "September 28"),
        ("Mei Tan", "Evergreen Transit", "Melbourne", "July 24"),
    ]
    for index, (person, org, location, date) in enumerate(names, start=1):
        prompt = f"Extract named entities: {person} from {org} met partners in {location} on {date}."
        check = {
            "type": "entity_set",
            "entities": [
                {"text": person, "type": "PERSON"},
                {"text": org, "type": "ORG"},
                {"text": location, "type": "LOCATION"},
                {"text": date, "type": "DATE"},
            ],
        }
        _add(rows, "ner", index, prompt, check, "person/org/location/date extraction")


def _code_debugging(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    items = [
        ("Debug this Python code and provide the corrected code: def max_value(items):\n    best = items[0]\n    for item in items:\n        return best\nIt should return the largest item.", ["def max_value", "return best"], "early return in max loop"),
        ("Fix the off-by-one bug: def count_down(n):\n    return list(range(n, 0, -1))\nIt should include 0 at the end.", ["range", "-1"], "range end"),
        ("Correct the mutable default: def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags", ["None", "tags"], "mutable default"),
        ("Debug this late-binding code: funcs=[]\nfor i in range(3):\n    funcs.append(lambda: i)\nEach lambda should return its own i.", ["lambda", "i=i"], "late binding closure"),
        ("Fix this generator exhaustion bug: nums = (n for n in [1, 2, 3])\nfirst = sum(nums)\nsecond = sum(nums)\nBoth sums should be 6.", ["list", "sum"], "generator exhausted"),
        ("Debug this sorting code: words=['pear','fig','apple']\nwords.sort()\nIt should sort by word length.", ["key=len", "sort"], "wrong sort key"),
        ("Fix the missing empty-list edge case: def average(nums):\n    return sum(nums)/len(nums)\nReturn 0 for an empty list.", ["if not nums", "0"], "empty input"),
        ("Correct this code: def is_even(n):\n    return n % 2\nIt should return True for even numbers.", ["== 0", "return"], "truthy remainder bug"),
        ("Debug: def reverse_words(words):\n    words.reverse()\nIt should return the reversed list.", ["return", "words"], "missing return"),
        ("Fix this accumulator bug: total=0\nfor n in [1,2,3]:\n    total=n\nprint(total)\nIt should print 6.", ["+=", "total"], "overwriting accumulator"),
        ("Correct this function: def first(items):\n    return items[0]\nIt should return None when the list is empty.", ["if", "None"], "empty edge case"),
        ("Fix this code: def clamp(x):\n    if x < 0: return 0\n    if x > 10: return x\n    return x\nValues above 10 should return 10.", ["return 10", "x > 10"], "upper bound"),
        ("Debug this dictionary count: counts={}\nfor word in words:\n    counts[word]=1\nIt should count repeated words.", ["get", "+ 1"], "count increment"),
        ("Correct this code: def last_even(nums):\n    for n in nums:\n        if n % 2 == 0:\n            return n\nIt should return the last even number.", ["reversed", "return n"], "first vs last"),
        ("Fix this Python code: def normalize(name):\n    name.strip().lower()\nIt should return the normalized string.", ["return", "strip", "lower"], "missing return chaining"),
        ("Debug this list copy bug: def add_item(items):\n    copy = items\n    copy.append('x')\n    return copy\nIt should not mutate the original list.", ["copy", "list"], "aliasing"),
        ("Correct this code: def contains_a(text):\n    return text.find('a')\nIt should return a boolean.", ["in text", "return"], "find integer vs boolean"),
        ("Fix this loop: for i in range(len(items)+1):\n    print(items[i])\nIt should not index past the end.", ["range(len(items))"], "off-by-one index"),
        ("Debug: def merge(a,b):\n    a.append(b)\n    return a\nIt should concatenate two lists.", ["extend", "return"], "append nested list"),
        ("Correct this code: def positive(nums):\n    return [n for n in nums if n >= 0]\nIt should include only numbers greater than zero.", ["> 0", "return"], "boundary condition"),
    ]
    for index, (prompt, values, notes) in enumerate(items, start=1):
        _add(rows, "code_debugging", index, prompt, {"type": "contains_all", "values": values}, notes)


def _logic(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    items = [
        ("Logic puzzle: Alice is older than Ben. Ben is older than Cara. Who is youngest?", "Cara", "ordering"),
        ("If all bronze keys open the archive and this key is bronze, does this key open the archive?", "yes", "syllogism"),
        ("Constraint puzzle: The red box is left of the blue box. The green box is right of the blue box. Which box is in the middle?", "blue", "linear order"),
        ("Exactly one of Nora and Omar took the cookie. Nora tells the truth and says Omar took it. Omar lies and says Nora took it. Who took it?", "Omar", "truth lie"),
        ("If no electric scooters are allowed in the lobby and Luna's vehicle is an electric scooter, is it allowed?", "no", "negative syllogism"),
        ("Seating puzzle: Ana sits left of Bo. Bo sits left of Cy. Who sits in the middle?", "Bo", "seating order"),
        ("Ownership puzzle: The red mug belongs to Kim. The blue mug belongs to Lee. Which mug belongs to Kim?", "red", "ownership"),
        ("Logic puzzle: Every raven in the story is black. This bird is a raven in the story. Is it black?", "yes", "deduction"),
        ("Constraint puzzle: The cat is not in box 1. It is not in box 3. Boxes are 1, 2, and 3. Which box has the cat?", "2", "elimination"),
        ("Truth puzzle: Pia always lies. Pia says the coin is heads. Is the coin heads?", "no", "liar"),
        ("Order puzzle: Dan finished after Eli but before Fay. Who finished first?", "Eli", "rank"),
        ("If all sealed jars are safe and this jar is not sealed, can we conclude it is safe?", "no", "invalid converse"),
        ("Constraint puzzle: The square is above the circle. The triangle is below the circle. Which shape is in the middle?", "circle", "vertical order"),
        ("Logic puzzle: Exactly one switch is on. Switch A is off. Switch B is off. Which switch is on: A, B, or C?", "C", "exclusion"),
        ("Logic puzzle: If every ticket with a star gets a prize and Maya's ticket has a star, does Maya get a prize?", "yes", "simple rule"),
        ("Seating puzzle: Jon is immediately right of Kai. Mira is immediately right of Jon. Who is leftmost?", "Kai", "seating"),
        ("Pet puzzle: The teacher owns the fish. Raul owns the turtle. Which pet does Raul own?", "turtle", "ownership"),
        ("If no guests under 18 can enter and Sam is 17, can Sam enter?", "no", "age rule"),
        ("Logic puzzle: One door is green and one is yellow. The prize is not behind the green door. Which door has the prize?", "yellow", "two options"),
        ("Truth puzzle: Tia tells the truth. Tia says Max has the map. Who has the map?", "Max", "truth teller"),
    ]
    for index, (prompt, label, notes) in enumerate(items, start=1):
        _add(rows, "logic", index, prompt, {"type": "contains_all", "values": [label]}, notes)


def _code_generation(rows: list[tuple[dict[str, str], dict[str, Any]]]) -> None:
    specs = [
        ("Write a Python function named second_largest(nums) that returns the second-largest distinct number, or None if it does not exist.", "second_largest", "assert second_largest([3, 1, 3, 2]) == 2\nassert second_largest([5]) is None\nassert second_largest([-1, -3, -2]) == -2"),
        ("Write a Python function named unique_ordered(items) that returns unique items while preserving first-seen order.", "unique_ordered", "assert unique_ordered([1,2,1,3,2]) == [1,2,3]\nassert unique_ordered([]) == []"),
        ("Write a Python function named count_words(words) that returns a dictionary of word counts.", "count_words", "assert count_words(['a','b','a']) == {'a':2,'b':1}\nassert count_words([]) == {}"),
        ("Write a Python function named parse_pairs(text) that parses 'a=1;b=2' into {'a':'1','b':'2'}.", "parse_pairs", "assert parse_pairs('a=1;b=2') == {'a':'1','b':'2'}\nassert parse_pairs('') == {}"),
        ("Write a Python function named first_or_default(items, default=None) that returns the first item or default when empty.", "first_or_default", "assert first_or_default([4,5]) == 4\nassert first_or_default([], 'x') == 'x'"),
        ("Write a Python function named flatten_once(items) that flattens one level of nested lists.", "flatten_once", "assert flatten_once([[1,2],[3],4]) == [1,2,3,4]\nassert flatten_once([]) == []"),
        ("Write a Python function named safe_ratio(a, b) that returns None when b is zero, otherwise a / b.", "safe_ratio", "assert safe_ratio(6,3) == 2\nassert safe_ratio(1,0) is None"),
        ("Write a Python function named clamp(x, low, high) that limits x to the inclusive range.", "clamp", "assert clamp(5,0,10) == 5\nassert clamp(-1,0,10) == 0\nassert clamp(12,0,10) == 10"),
        ("Write a Python function named chunk_pairs(items) that returns consecutive pairs, dropping a final unpaired item.", "chunk_pairs", "assert chunk_pairs([1,2,3,4,5]) == [(1,2),(3,4)]\nassert chunk_pairs([]) == []"),
        ("Write a Python function named initials(name) that returns uppercase initials from a full name.", "initials", "assert initials('Ada Lovelace') == 'AL'\nassert initials('  grace  hopper ') == 'GH'"),
        ("Write a Python function named moving_sum(nums) that returns cumulative sums.", "moving_sum", "assert moving_sum([2,3,5]) == [2,5,10]\nassert moving_sum([]) == []"),
        ("Write a Python function named invert_dict(d) that maps values to lists of keys preserving insertion order.", "invert_dict", "assert invert_dict({'a':1,'b':1,'c':2}) == {1:['a','b'],2:['c']}"),
        ("Write a Python function named is_palindrome(text) that ignores case and spaces.", "is_palindrome", "assert is_palindrome('Never odd or even') is True\nassert is_palindrome('hello') is False"),
        ("Write a Python function named only_positive(nums) that returns numbers greater than zero.", "only_positive", "assert only_positive([-1,0,2,3]) == [2,3]"),
        ("Write a Python function named merge_counts(a, b) that adds values from two dictionaries.", "merge_counts", "assert merge_counts({'x':2},{'x':3,'y':1}) == {'x':5,'y':1}"),
        ("Write a Python function named last_index(items, value) that returns the last index of value or -1.", "last_index", "assert last_index([1,2,1],1) == 2\nassert last_index([1],9) == -1"),
        ("Write a Python function named title_names(names) that strips and title-cases each name.", "title_names", "assert title_names([' ada ','GRACE']) == ['Ada','Grace']"),
        ("Write a Python function named has_duplicates(items) that returns True if any item appears more than once.", "has_duplicates", "assert has_duplicates([1,2,1]) is True\nassert has_duplicates([1,2,3]) is False"),
        ("Write a Python function named median_of_three(a, b, c) that returns the middle value.", "median_of_three", "assert median_of_three(3,1,2) == 2\nassert median_of_three(9,9,1) == 9"),
        ("Write a Python function named compact_none(items) that removes None values while keeping other falsey values.", "compact_none", "assert compact_none([None,0,'',2]) == [0,'',2]"),
    ]
    for index, (prompt, function_name, tests) in enumerate(specs, start=1):
        _add(
            rows,
            "code_generation",
            index,
            prompt,
            {"type": "python_exec", "function": function_name, "tests": tests},
            "executable edge-case function test",
        )


def build_rows() -> list[tuple[dict[str, str], dict[str, Any]]]:
    rows: list[tuple[dict[str, str], dict[str, Any]]] = []
    for builder in (
        _factual,
        _math,
        _sentiment,
        _summarization,
        _ner,
        _code_debugging,
        _logic,
        _code_generation,
    ):
        builder(rows)
    return rows


def main() -> int:
    rows = build_rows()
    tasks = [task for task, _ in rows]
    expected = [item for _, item in rows]
    TASKS_PATH.write_text(json.dumps(tasks, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    EXPECTED_PATH.write_text(json.dumps(expected, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {len(tasks)} tasks to {TASKS_PATH}")
    print(f"wrote {len(expected)} expected rows to {EXPECTED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
