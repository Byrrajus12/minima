from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minima.answer_validation import (
    classify_code_generation_risk,
    detect_truncation,
    derive_prompt_code_examples,
    has_mutable_default_defect,
    normalize_answer,
    parse_ner_lines,
    strip_code_fences,
    summary_requirements,
    validate_answer,
)
from minima.config import Config
from minima.fireworks_client import FireworksClient
from minima.local_llm import LocalGeneration
from minima.main import write_results
from minima.runtime import LocalFirstOrchestrator, deterministic_answer
from minima.task_classifier import classify_task


class RuntimePipelineTests(unittest.TestCase):
    def test_classification_core_categories(self) -> None:
        self.assertEqual(classify_task("Give a sentiment label and one reason: loved it but setup was bad."), "sentiment")
        self.assertEqual(classify_task("Extract named entities: Maya joined Acme in Paris today."), "ner")
        self.assertEqual(classify_task("Write foo(x) returning x + 1."), "code_generation")
        self.assertEqual(classify_task("A sits immediately left of B. Who is first?"), "logic")

    def test_math_contradiction_rejected(self) -> None:
        answer = normalize_answer("math", "FINAL: usable filters = 400\nCHECK: 480 - 135 + 192 - 17 = 520. Wait, correction needed")
        result = validate_answer("math", "How many filters remain?", answer)
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_code, "correction_language")

    def test_math_approximate_exact_rejected(self) -> None:
        answer = normalize_answer("math", "FINAL: exactly 450 liters")
        result = validate_answer("math", "The tank is about half full with capacity 900 liters. How many liters exactly?", answer)
        self.assertFalse(result.valid)

    def test_logic_ordered_assignment_rejected(self) -> None:
        prompt = "Kira is in seat 4. Ivo sits immediately left of Jia. Luis is not adjacent to Kira. Give the order."
        answer = normalize_answer("logic", "ANSWER: 1-Ivo, 2-Jia, 3-Luis, 4-Kira")
        result = validate_answer("logic", prompt, answer)
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_code, "ordered_assignment_violation")

    def test_ner_exact_span_and_merged_entity_rejected(self) -> None:
        prompt = "Extract entities: The Open River Project named Elena GarcÃ­a in Porto's Casa da MÃºsica on May 9."
        merged = normalize_answer("ner", "Open River Project | ORG\nElena GarcÃ­a | PERSON\nPorto's Casa da MÃºsica | LOCATION\nMay 9 | DATE")
        self.assertFalse(validate_answer("ner", prompt, merged).valid)
        titled = normalize_answer("ner", "Dr. Amara Okafor | PERSON", "Extract entities: Dr. Amara Okafor spoke.")
        self.assertEqual(titled.answer, "Amara Okafor | PERSON")
        self.assertTrue(validate_answer("ner", "Extract entities: Dr. Amara Okafor spoke.", titled).valid)

    def test_sentiment_label_and_reason(self) -> None:
        good = normalize_answer("sentiment", "LABEL: mixed\nREASON: It praises speed but criticizes setup.")
        self.assertTrue(validate_answer("sentiment", "Give a label and one reason.", good).valid)
        bad = normalize_answer("sentiment", "LABEL: delighted\nREASON: good")
        self.assertFalse(validate_answer("sentiment", "Give a label and one reason.", bad).valid)

    def test_summary_structure(self) -> None:
        self.assertEqual(summary_requirements("Summarize in exactly two bullets.")["bullets"], 2)
        bad = normalize_answer("summarization", "One sentence only.")
        self.assertFalse(validate_answer("summarization", "Summarize in exactly two bullets.", bad).valid)

    def test_code_fence_removal_and_name_validation(self) -> None:
        code, stripped = strip_code_fences("```python\ndef foo(x):\n    return x\n```")
        self.assertTrue(stripped)
        normalized = normalize_answer("code_generation", code)
        self.assertTrue(validate_answer("code_generation", "Write foo(x) returning x.", normalized).valid)

    def test_one_repair_maximum_and_offline_fallback(self) -> None:
        config = Config(None, None, (), True)
        orchestrator = LocalFirstOrchestrator(FireworksClient(config))
        generations = [
            LocalGeneration("LABEL: delighted", "LABEL: delighted", 1, 1, 2, 1.0, False),
            LocalGeneration("LABEL: positive\nREASON: clear praise", "LABEL: positive\nREASON: clear praise", 1, 1, 2, 1.0, False),
        ]

        def fake_generate(**_: object) -> LocalGeneration:
            return generations.pop(0)

        with patch("minima.runtime.local_generate_with_metadata", side_effect=fake_generate):
            trace = orchestrator.answer_with_trace("Give a sentiment label and one reason: I loved it.", "t1")
        self.assertEqual(trace.repair_calls, 1)
        self.assertTrue(trace.repair_success)
        self.assertEqual(trace.fireworks_calls, 0)

    def test_results_schema_exact(self) -> None:
        path = Path("tests_tmp_results.json")
        try:
            write_results(path, [{"task_id": "a", "answer": "b", "extra": "removed"}])
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data[0]), {"task_id", "answer"})
        finally:
            if path.exists():
                path.unlink()

    def test_saved_protocol_v3_regressions_rejected(self) -> None:
        ledger = Path("policy_eval_artifacts/model_compare/qwen3_4b_protocol_v3/ledger.json")
        if not ledger.exists():
            self.skipTest("protocol-v3 ledger absent")
        rows = {row["task_id"]: row for row in json.loads(ledger.read_text(encoding="utf-8"))["rows"]}
        for task_id in ("vh_math_02", "vh_math_08", "vh_logic_02", "vh_logic_04", "vh_ner_05"):
            row = rows[task_id]
            normalized = normalize_answer(row["expected_category"], row["answer"], row["task_prompt"])
            result = validate_answer(row["expected_category"], row["task_prompt"], normalized)
            self.assertFalse(result.valid, task_id)
        math10 = rows["vh_math_10"]
        result = validate_answer("math", math10["task_prompt"], normalize_answer("math", math10["answer"]))
        self.assertTrue(result.valid)

    def test_parse_ner_lines(self) -> None:
        self.assertEqual(parse_ner_lines("Maya | PERSON\nAcme | ORG"), [("Maya", "PERSON"), ("Acme", "ORG")])

    def test_truncation_complete_and_incomplete_at_cap(self) -> None:
        complete_sentiment = normalize_answer("sentiment", "LABEL: neutral\nREASON: The statement has no positive or negative opinion.")
        sentiment_trunc = detect_truncation("sentiment", "Give a label and reason.", complete_sentiment, completion_tokens=48, max_completion_tokens=48)
        self.assertTrue(sentiment_trunc.possibly_truncated)
        self.assertFalse(sentiment_trunc.incomplete)

        incomplete_fact = normalize_answer("factual", "The checksum differs when data has been altered or")
        fact_trunc = detect_truncation("factual", "What does a checksum do?", incomplete_fact, finish_reason="length", completion_tokens=96, max_completion_tokens=96)
        self.assertTrue(fact_trunc.incomplete)
        self.assertFalse(validate_answer("factual", "What does a checksum do?", incomplete_fact, truncation=fact_trunc).valid)

        math = normalize_answer("math", "FINAL: 42\nExplanation: because the")
        math_trunc = detect_truncation("math", "Compute.", math, finish_reason="length", completion_tokens=128, max_completion_tokens=128)
        self.assertTrue(validate_answer("math", "Compute.", math, truncation=math_trunc).valid)

    def test_cap_hit_prose_ending_mid_sentence_is_rejected(self) -> None:
        answer = normalize_answer("factual", "The sensor compares the current reading with the calibrated baseline and then")
        trunc = detect_truncation("factual", "Explain what the sensor does.", answer, finish_reason="length", completion_tokens=96, max_completion_tokens=96)
        self.assertIn("cap_hit_unfinished_prose", trunc.reasons)
        self.assertTrue(trunc.incomplete)
        self.assertEqual(validate_answer("factual", "Explain what the sensor does.", answer, truncation=trunc).failure_code, "truncated_output")

    def test_cap_hit_complete_prose_conclusion_is_retained(self) -> None:
        answer = normalize_answer("factual", "The sensor flags a mismatch when the current reading no longer matches the calibrated baseline.")
        trunc = detect_truncation("factual", "Explain what the sensor does.", answer, finish_reason="length", completion_tokens=96, max_completion_tokens=96)
        self.assertTrue(trunc.possibly_truncated)
        self.assertFalse(trunc.incomplete)
        self.assertTrue(validate_answer("factual", "Explain what the sensor does.", answer, truncation=trunc).valid)

    def test_stop_finished_prose_remains_unchanged(self) -> None:
        answer = normalize_answer("factual", "The sensor compares readings with its baseline and raises an alert")
        trunc = detect_truncation("factual", "Explain what the sensor does.", answer, finish_reason="stop", completion_tokens=12, max_completion_tokens=96)
        self.assertFalse(trunc.possibly_truncated)
        self.assertTrue(validate_answer("factual", "Explain what the sensor does.", answer, truncation=trunc).valid)

    def test_incomplete_python_code_at_cap(self) -> None:
        code = normalize_answer("code_generation", "def add_one(x):\n    return x +")
        trunc = detect_truncation("code_generation", "Write add_one(x).", code, completion_tokens=256, max_completion_tokens=256)
        self.assertTrue(trunc.incomplete)
        self.assertEqual(validate_answer("code_generation", "Write add_one(x).", code, truncation=trunc).failure_code, "truncated_output")

    def test_ner_completeness_patterns_and_relative_dates(self) -> None:
        prompt = "Extract named entities: Washington joined Meridian as counsel in Washington, D.C., on Monday."
        missing_org = normalize_answer("ner", "Washington | PERSON\nWashington, D.C. | LOCATION\nMonday | DATE", prompt)
        result = validate_answer("ner", prompt, missing_org)
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_code, "missing_required_entity")
        complete = normalize_answer("ner", "Washington | PERSON\nMeridian | ORG\nWashington, D.C. | LOCATION\nMonday | DATE", prompt)
        self.assertTrue(validate_answer("ner", prompt, complete).valid)

    def test_finite_domain_logic_unique_nonunique_and_contradiction(self) -> None:
        nonunique = deterministic_answer("logic", "A, B, and C each choose tea or coffee; choices need not differ. A chooses tea. B chooses the same as C. What does B choose?")
        self.assertIn("Cannot be determined uniquely", nonunique or "")
        unique = deterministic_answer("logic", "Mia, Noa, and Pia each choose red or blue. Mia chooses red. Noa chooses the same as Mia. What does Noa choose?")
        self.assertIn("Noa chooses red", unique or "")
        impossible = deterministic_answer("logic", "Rae and Sol each choose tea or coffee. Rae chooses tea. Rae does not choose tea. What does Rae choose?")
        self.assertIn("No valid assignment", impossible or "")

    def test_mutable_default_ast_checking(self) -> None:
        import ast

        self.assertTrue(has_mutable_default_defect(ast.parse("def f(x, seen=[]):\n    return seen")))
        self.assertTrue(has_mutable_default_defect(ast.parse("def f(x, cache=dict()):\n    return cache")))
        self.assertTrue(has_mutable_default_defect(ast.parse("def f(x, flags=set()):\n    return flags")))
        self.assertFalse(has_mutable_default_defect(ast.parse("def f(x, seen=None):\n    return seen")))
        self.assertFalse(has_mutable_default_defect(ast.parse("def f(x, seen=()):\n    return seen")))
        bad = normalize_answer("code_debugging", "def remember(x, seen=[]):\n    seen.append(x)\n    return seen")
        result = validate_answer("code_debugging", "Fix the mutable default bug. ```python\ndef remember(x, seen=[]):\n    return seen\n```", bad)
        self.assertEqual(result.failure_code, "mutable_default_still_present")

    def test_code_generation_risk_and_prompt_examples(self) -> None:
        self.assertEqual(classify_code_generation_risk("Write count_even(values) returning the number of even integers.")["risk"], "low")
        self.assertEqual(classify_code_generation_risk("Write rotate(values, steps) returning a new list rotated right and leave input unchanged.")["risk"], "high")
        examples = derive_prompt_code_examples("Write double(x). double(3) should return 6.", "double")
        self.assertEqual(examples, [("double(3)", 6)])
        good = normalize_answer("code_generation", "def double(x):\n    return x * 2")
        self.assertTrue(validate_answer("code_generation", "Write double(x). double(3) should return 6.", good).valid)
        bad = normalize_answer("code_generation", "def double(x):\n    return x + 2")
        self.assertEqual(validate_answer("code_generation", "Write double(x). double(3) should return 6.", bad).failure_code, "prompt_example_failed")

    def test_valid_sentiment_reason_words_no_repair(self) -> None:
        config = Config(None, None, (), True)
        orchestrator = LocalFirstOrchestrator(FireworksClient(config))
        generation = LocalGeneration(
            "LABEL: neutral\nREASON: The statement contains no positive or negative opinion.",
            "LABEL: neutral\nREASON: The statement contains no positive or negative opinion.",
            1,
            48,
            49,
            1.0,
            False,
            max_completion_tokens=48,
            reached_token_cap=True,
        )
        with patch("minima.runtime.local_generate_with_metadata", return_value=generation):
            trace = orchestrator.answer_with_trace("Give a sentiment label and one reason: The port uses TCP.", "sent1")
        self.assertEqual(trace.repair_calls, 0)
        self.assertTrue(trace.accepted_local)

    def test_fireworks_fallback_decisions_without_calling_fireworks(self) -> None:
        config = Config(None, None, (), True)
        orchestrator = LocalFirstOrchestrator(FireworksClient(config))
        generation = LocalGeneration("def rotate(values, steps):\n    return values", "def rotate(values, steps):\n    return values", 1, 4, 5, 1.0, False, max_completion_tokens=256)
        with patch("minima.runtime.local_generate_with_metadata", return_value=generation):
            trace = orchestrator.answer_with_trace("Write rotate(values, steps) returning a new list rotated right. Support negative steps and leave input unchanged.", "code1")
        self.assertTrue(trace.would_fallback)
        self.assertEqual(trace.fireworks_calls, 0)

    def test_production_entrypoint_smoke_without_qwen(self) -> None:
        from minima.main import run

        input_path = Path("tests_tmp_tasks.json")
        output_path = Path("tests_tmp_output.json")
        try:
            input_path.write_text(json.dumps([{"task_id": "m1", "prompt": "What is 2 plus 3?"}]), encoding="utf-8")
            with patch.dict("os.environ", {"FIREWORKS_API_KEY": "", "FIREWORKS_BASE_URL": "", "ALLOWED_MODELS": "", "MINIMA_LOCAL_MODEL_PATH": "Z:/no/model.gguf"}):
                self.assertEqual(run(input_path, output_path), 0)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(set(data[0]), {"task_id", "answer"})
        finally:
            for path in (input_path, output_path):
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
