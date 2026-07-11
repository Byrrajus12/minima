from policy_eval.scoring import score_answer


def test_numeric_accepts_single_value_schema() -> None:
    assert score_answer("The answer is 64.8.", {"type": "numeric", "value": 64.8}).passed


def test_numeric_accepts_values_schema() -> None:
    check = {"type": "numeric", "values": [64.8, 240]}
    assert score_answer("64.8 grams and 240 ml", check).passed


def test_label_accepts_value_schema() -> None:
    assert score_answer("positive", {"type": "label", "value": "positive"}).passed


def test_contains_all_schema() -> None:
    check = {"type": "contains_all", "values": ["mitochond", "oxidative phosphorylation"]}
    assert score_answer("Mitochondria perform oxidative phosphorylation.", check).passed


def test_summary_accepts_required_and_sentences_schema() -> None:
    check = {"type": "summary", "required": ["closed", "Monday", "boiler"], "sentences": 1}
    assert score_answer("Closed Monday because of the boiler.", check).passed


def test_entities_accepts_values_pair_schema() -> None:
    check = {
        "type": "entities",
        "values": [["Dr. Maya Chen", "PERSON"], ["Lagos", "LOCATION"]],
    }
    assert score_answer("Dr. Maya Chen - PERSON; Lagos - LOCATION", check).passed


def test_python_function_schema_executes_tests() -> None:
    answer = "def sum_positive(values):\n    return sum(v for v in values if v > 0)"
    check = {
        "type": "python_function",
        "name": "sum_positive",
        "tests": ["assert sum_positive([-2, 3, 4]) == 7"],
    }
    assert score_answer(answer, check).passed


def test_python_function_schema_allows_standard_exceptions_in_tests() -> None:
    answer = "def require_value(value):\n    if value is None:\n        raise ValueError('missing')\n    return value"
    check = {
        "type": "python_function",
        "name": "require_value",
        "tests": [
            "try:\n    require_value(None)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('expected ValueError')"
        ],
    }
    assert score_answer(answer, check).passed


def test_escalate_schema_accepts_non_empty_answer() -> None:
    assert score_answer("Needs manual review.", {"type": "escalate"}).passed
