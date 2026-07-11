from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from policy_eval import evaluate_holdout_48 as holdout


class HoldoutEvaluatorPreparationTests(unittest.TestCase):
    def _synthetic_run_context(self, tmp: str, count: int = 3) -> tuple[Path, Path, Path, Path]:
        base = Path(tmp)
        tasks = [{"task_id": f"t{i}", "category": "math", "prompt": f"Task {i}"} for i in range(1, count + 1)]
        expected = [{"task_id": f"t{i}", "check": {"type": "numeric", "value": i}} for i in range(1, count + 1)]
        tasks_path = base / "tasks.json"
        expected_path = base / "expected.json"
        manifest_path = base / "manifest.json"
        tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
        expected_path.write_text(json.dumps(expected), encoding="utf-8")
        manifest_path.write_text(json.dumps({"tasks": []}), encoding="utf-8")
        artifact = base / "artifacts"
        artifact.mkdir()
        model = base / "model.gguf"
        model.write_bytes(b"fake-model")
        holdout.atomic_write_json(artifact / "prepared_one_shot_manifest.json", {"status": "prepared", "model_hash": hashlib.sha256(b"fake-model").hexdigest()})
        holdout.atomic_write_json(artifact / "production_hash_manifest.json", {"files": {}, "manifest_hash": "x"})
        return tasks_path, expected_path, manifest_path, artifact

    def _run_args(self, artifact: Path, output: Path | None = None) -> Namespace:
        model = artifact.parent / "model.gguf"
        return Namespace(
            model_path=str(model),
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            artifact_dir=str(artifact),
            output_dir=str(output or artifact / "exec"),
        )

    def _patch_run_globals(self, tasks_path: Path, expected_path: Path, manifest_path: Path):
        return patch.multiple(holdout, TASKS_PATH=tasks_path, EXPECTED_PATH=expected_path, MANIFEST_PATH=manifest_path)

    def _fake_trace(self, task_id: str, answer: str):
        generation = SimpleNamespace(raw_text=answer, finish_reason="stop", max_completion_tokens=5, completion_tokens=1, reached_token_cap=False, runtime_ms=1.0)
        validation = SimpleNamespace(valid=True, failure_code="", failure_detail="")
        return SimpleNamespace(
            answer=answer,
            category="math",
            path="local_primary",
            deterministic_attempted=False,
            deterministic_answer=None,
            generation=generation,
            repair_generation=None,
            normalized=SimpleNamespace(answer=answer),
            validation=validation,
            primary_truncation=SimpleNamespace(incomplete=False),
            repair_calls=0,
            repair_normalized=None,
            repair_validation=None,
            fallback_reason="",
            prompt_tokens=1,
            completion_tokens=1,
            accepted_local=True,
            unverified_local=False,
            first_pass_valid=True,
            final_valid=True,
            actual_fallback=False,
            would_fallback=False,
            fireworks_calls=0,
        )

    def _fake_fireworks_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            placeholder_mode=False,
            model="kimi-k2p7-code",
            fireworks_base_url="http://fireworks.test",
            fireworks_api_key="secret",
            request_timeout_seconds=5,
        )

    def _urlopen_response(self, payload: dict[str, object]):
        class Response:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def read(self_inner):
                return json.dumps(payload).encode("utf-8")

        return Response()

    def _http_error_body(self, body: bytes):
        return SimpleNamespace(read=lambda: body, close=lambda: None)

    def test_holdout_metadata_validation(self) -> None:
        metadata = holdout.check_holdout_metadata()
        self.assertEqual(metadata["task_count"], 48)
        self.assertEqual(metadata["unique_task_id_count"], 48)
        self.assertEqual(metadata["task_count_per_category"], holdout.EXPECTED_COUNTS)
        self.assertTrue(metadata["manifest_counts_agree"])
        self.assertEqual(metadata["difficulty_count"], {"easy": 16, "hard": 16, "medium": 16})
        self.assertTrue(all(counts == {"easy": 2, "hard": 2, "medium": 2} for counts in metadata["difficulty_count_per_category"].values()))
        self.assertTrue(metadata["all_tasks_have_expected_checks"])
        self.assertTrue(metadata["composition_valid"])

    def test_overlap_audit_uses_synthetic_fixtures(self) -> None:
        holdout_tasks = [
            {"task_id": "h1", "category": "math", "prompt": "A shop has 10 pens and buys 5 more. How many pens?"},
            {"task_id": "h2", "category": "factual", "prompt": "Explain why metal feels colder than wood."},
        ]
        dev_tasks = [
            {"task_id": "d1", "category": "math", "prompt": "A shop has 20 pens and buys 7 more. How many pens?"},
            {"task_id": "d2", "category": "logic", "prompt": "Who sits first if Ana is before Bo?"},
        ]
        audit = holdout.overlap_audit(holdout_tasks, dev_tasks)
        self.assertEqual(audit["exact_duplicate_count"], 0)
        self.assertIn("maximum_observed_similarity", audit)

    def test_structural_overlap_replaces_names_numbers_and_dates(self) -> None:
        a = holdout.structural_prompt("Maya met Acme Labs on May 9, 2025 and bought 42 filters.")
        b = holdout.structural_prompt("Noor met Horizon Works on June 7, 2024 and bought 13 filters.")
        c = holdout.structural_prompt("Explain why metal conducts heat better than wood.")
        self.assertGreaterEqual(__import__("difflib").SequenceMatcher(None, a, b).ratio(), 0.9)
        self.assertLess(__import__("difflib").SequenceMatcher(None, a, c).ratio(), 0.75)

    def test_production_hash_manifest_and_mismatch_refusal(self) -> None:
        manifest = holdout.production_hash_manifest()
        normalized_keys = {key.replace("\\", "/") for key in manifest["files"]}
        self.assertIn("src/minima/main.py", normalized_keys)
        self.assertTrue(all("/" in key and "\\" not in key for key in manifest["files"]))
        ok, mismatches = holdout.verify_hash_manifest(manifest)
        self.assertTrue(ok)
        self.assertEqual(mismatches, [])
        tampered = json.loads(json.dumps(manifest))
        main_key = next(key for key in tampered["files"] if key.replace("\\", "/") == "src/minima/main.py")
        tampered["files"][main_key] = "0" * 64
        ok, mismatches = holdout.verify_hash_manifest(tampered)
        self.assertFalse(ok)
        self.assertTrue(any("src/minima/main.py" in item.replace("\\", "/") for item in mismatches))
        bad_path = {"files": {"../file": "0" * 64}}
        ok, mismatches = holdout.verify_hash_manifest(bad_path)
        self.assertFalse(ok)
        self.assertTrue(any("unsafe" in item or "escapes" in item for item in mismatches))

    def test_expected_answer_isolation_before_scoring(self) -> None:
        source = Path("policy_eval/evaluate_holdout_48.py").read_text(encoding="utf-8")
        generation_start = source.index("tasks = load_json(TASKS_PATH)")
        scoring_start = source.index("expected = load_json(EXPECTED_PATH)")
        self.assertLess(generation_start, scoring_start)

    def test_fixed_task_order_and_checkpoint_interval_are_declared(self) -> None:
        source = Path("policy_eval/evaluate_holdout_48.py").read_text(encoding="utf-8")
        self.assertIn("enumerate(tasks, start=1)", source)
        self.assertIn("index % 8 == 0", source)

    def test_checkpoint_and_output_schema(self) -> None:
        row = holdout.production_result_row("x", "answer")
        self.assertEqual(set(row), {"task_id", "answer"})
        self.assertEqual(holdout.ledger_row_schema()[0], "task_id")
        self.assertIn("result_keys", holdout.ledger_row_schema())

    def test_one_shot_lock_behavior_and_no_bypass_option(self) -> None:
        metadata = holdout.check_holdout_metadata()
        prod = holdout.production_hash_manifest()
        prepared = holdout.make_prepared_manifest(metadata, prod, "a" * 64)
        self.assertEqual(prepared["status"], "prepared")
        self.assertFalse(prepared["rules"]["force_option_available"])
        source = Path("policy_eval/evaluate_holdout_48.py").read_text(encoding="utf-8")
        self.assertNotIn("--force", source)
        self.assertNotIn("--unlock", source)
        self.assertNotIn("--reset", source)

    def test_automated_score_freezing_and_semantic_audit_separation(self) -> None:
        source = Path("policy_eval/evaluate_holdout_48.py").read_text(encoding="utf-8")
        run_source = source[source.index("def run_holdout") : source.index("def main")]
        strict_write = run_source.index('"strict_automated_report.json"')
        strict_hash = run_source.index("strict_hash = sha256_file(strict_path)")
        semantic = run_source.index('"semantic_audit_queue.json"')
        self.assertLess(strict_write, strict_hash)
        self.assertLess(strict_hash, semantic)

    def test_readiness_threshold_logic(self) -> None:
        good = {
            "strict_final_score": 43,
            "deterministic_solver_wrong_count": 0,
            "verified_local_objectively_wrong_count": 1,
            "repair_regression_count": 0,
            "malformed_output_count": 0,
            "missing_output_count": 0,
            "duplicate_output_count": 0,
            "runtime_failure_count": 0,
            "api_failure_count": 0,
            "wall_time_seconds": 599,
            "projected_19_task_runtime_seconds": 237.10416666666666,
            "fireworks_call_count": 9,
        }
        self.assertEqual(holdout.readiness_decision(good, True), "READY_TO_SUBMIT_CANDIDATE")
        bad = dict(good, strict_final_score=42)
        self.assertEqual(holdout.readiness_decision(bad, True), "NOT_READY_TO_SUBMIT")

    def test_prepare_import_does_not_import_production_runtime(self) -> None:
        source = Path("policy_eval/evaluate_holdout_48.py").read_text(encoding="utf-8")
        prep_source = source[source.index("def prepare") : source.index("def run_holdout")]
        self.assertNotIn("from minima.", prep_source)
        self.assertNotIn("import minima.", prep_source)

    def test_no_fireworks_or_model_during_preparation_tests(self) -> None:
        source = Path("policy_eval/evaluate_holdout_48.py").read_text(encoding="utf-8")
        prep_source = source[source.index("def prepare") : source.index("def run_holdout")]
        self.assertNotIn("llama_cpp", prep_source)
        self.assertNotIn("FireworksClient", prep_source)

    def test_absolute_minima_imports_are_followed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "src" / "minima"
            pkg.mkdir(parents=True)
            main = pkg / "main.py"
            helper = pkg / "helper.py"
            nested = pkg / "nested.py"
            main.write_text("from minima.helper import thing\nimport minima.nested\n", encoding="utf-8")
            helper.write_text("thing = 1\n", encoding="utf-8")
            nested.write_text("value = 2\n", encoding="utf-8")
            old_src = holdout.SRC
            try:
                holdout.SRC = root / "src"
                found = {p.name for p in holdout.relative_minima_imports(main)}
            finally:
                holdout.SRC = old_src
        self.assertIn("helper.py", found)
        self.assertIn("nested.py", found)

    def test_prepared_manifest_contains_all_frozen_hashes(self) -> None:
        metadata = holdout.check_holdout_metadata()
        prod = holdout.production_hash_manifest()
        prepared = holdout.make_prepared_manifest(metadata, prod, "b" * 64)
        self.assertIn("holdout_manifest_hash", prepared)
        self.assertEqual(prepared["status"], "prepared")

    def test_fireworks_allowed_models_short_and_full_paths(self) -> None:
        short = holdout.parse_and_validate_allowed_models("minimax-m3,kimi-k2p7-code")
        full = holdout.parse_and_validate_allowed_models("accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code")
        self.assertEqual(len(short), 2)
        self.assertEqual(len(full), 2)

    def test_fireworks_allowed_model_failures(self) -> None:
        with self.assertRaises(SystemExit):
            holdout.parse_and_validate_allowed_models("minimax-m3")
        with self.assertRaises(SystemExit):
            holdout.parse_and_validate_allowed_models("kimi-k2p7-code")
        with self.assertRaises(SystemExit):
            holdout.parse_and_validate_allowed_models("minimax-m3,,kimi-k2p7-code")

    def test_validate_preflight_mismatches(self) -> None:
        metadata = holdout.check_holdout_metadata()
        prod = holdout.production_hash_manifest()
        prepared = holdout.make_prepared_manifest(metadata, prod, holdout.sha256_file(Path("policy_eval/evaluate_holdout_48.py")))
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"fake-model")
            args = Namespace(model_path=str(model), model_sha256=hashlib.sha256(b"fake-model").hexdigest())
            with patch.dict(os.environ, {"FIREWORKS_API_KEY": "x", "FIREWORKS_BASE_URL": "http://127.0.0.1", "ALLOWED_MODELS": "minimax-m3,kimi-k2p7-code"}):
                holdout.validate_preflight(args, prepared, prod, Path(tmp) / "empty")
            with patch.dict(os.environ, {"FIREWORKS_API_KEY": "x", "FIREWORKS_BASE_URL": "http://127.0.0.1", "ALLOWED_MODELS": "minimax-m3,kimi-k2p7-code"}):
                with self.assertRaises(SystemExit):
                    holdout.validate_preflight(Namespace(model_path=str(model), model_sha256="0" * 64), prepared, prod, Path(tmp) / "empty2")
            bad_prepared = dict(prepared, evaluator_hash="0" * 64)
            with patch.dict(os.environ, {"FIREWORKS_API_KEY": "x", "FIREWORKS_BASE_URL": "http://127.0.0.1", "ALLOWED_MODELS": "minimax-m3,kimi-k2p7-code"}):
                with self.assertRaises(SystemExit):
                    holdout.validate_preflight(args, bad_prepared, prod, Path(tmp) / "empty3")
            bad_prod = json.loads(json.dumps(prod))
            bad_prod["manifest_hash"] = "0" * 64
            with patch.dict(os.environ, {"FIREWORKS_API_KEY": "x", "FIREWORKS_BASE_URL": "http://127.0.0.1", "ALLOWED_MODELS": "minimax-m3,kimi-k2p7-code"}):
                with self.assertRaises(SystemExit):
                    holdout.validate_preflight(args, prepared, bad_prod, Path(tmp) / "empty4")
            bad_manifest = dict(prepared, holdout_manifest_hash="0" * 64)
            with patch.dict(os.environ, {"FIREWORKS_API_KEY": "x", "FIREWORKS_BASE_URL": "http://127.0.0.1", "ALLOWED_MODELS": "minimax-m3,kimi-k2p7-code"}):
                with self.assertRaises(SystemExit):
                    holdout.validate_preflight(args, bad_manifest, prod, Path(tmp) / "empty5")

    def test_instrumented_fireworks_usage_capture(self) -> None:
        class FakeInner:
            config = SimpleNamespace(placeholder_mode=False)

            def answer(self, **_: object) -> str:
                return "remote answer"

        wrapped = holdout.InstrumentedFireworksClient(FakeInner())
        self.assertEqual(wrapped.answer(prompt="p", category="math", model="kimi", task_id="t1"), "remote answer")
        self.assertEqual(len(wrapped.calls), 1)
        self.assertFalse(wrapped.calls[0]["usage_known"])
        self.assertEqual(holdout.parse_fireworks_usage(json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}))["total_tokens"], 7)

    def test_instrumented_fireworks_failure_capture(self) -> None:
        class FakeInner:
            config = SimpleNamespace(placeholder_mode=False)

            def answer(self, **_: object) -> str:
                raise RuntimeError("boom")

        wrapped = holdout.InstrumentedFireworksClient(FakeInner())
        with self.assertRaises(RuntimeError):
            wrapped.answer(prompt="p", category="math", model="kimi", task_id="t1")
        self.assertFalse(wrapped.calls[0]["success"])

    def test_first_pass_and_final_scores_can_differ(self) -> None:
        results = [{"task_id": "t1", "answer": "42"}]
        expected = [{"task_id": "t1", "check": {"type": "numeric", "value": 42}}]
        ledger = [{"task_id": "t1", "category": "math", "selected_path": "local_repair", "local_normalized_answer": "41", "repair_answer": "42", "repair_attempt": True}]
        summary = holdout.summarize_scores(results, expected, ledger)
        self.assertEqual(summary["strict_first_pass_score"], 0)
        self.assertEqual(summary["strict_final_score"], 1)
        self.assertEqual(summary["repair_transitions"]["improved"], 1)

    def test_repair_and_remote_transitions_are_separate(self) -> None:
        results = [{"task_id": "t1", "answer": "42"}]
        expected = [{"task_id": "t1", "check": {"type": "numeric", "value": 42}}]
        ledger = [{
            "task_id": "t1",
            "category": "math",
            "selected_path": "remote",
            "local_normalized_answer": "40",
            "repair_answer": "41",
            "repair_attempt": True,
            "remote_call_count": 1,
        }]
        summary = holdout.summarize_scores(results, expected, ledger)
        self.assertEqual(summary["repair_transitions"]["unchanged_incorrect"], 1)
        self.assertEqual(summary["remote_transitions"]["improved"], 1)

    def test_routing_report_uses_trace_acceptance_fields(self) -> None:
        ledger = [
            {"task_id": "verified", "category": "math", "accepted_local": True, "unverified_local": False, "selected_path": "local_primary", "remote_call_count": 0},
            {"task_id": "unverified", "category": "math", "accepted_local": True, "unverified_local": True, "selected_path": "local_unverified_best_effort", "remote_call_count": 0},
        ]
        report = holdout.routing_report(ledger, [])
        self.assertEqual(report["verified_local_accept_count"], 1)
        self.assertEqual(report["unverified_local_count"], 1)

    def test_production_fireworks_records_one_logical_call_for_retry(self) -> None:
        success = {
            "choices": [{"message": {"content": "remote answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        http_error = urllib.error.HTTPError(
            "http://fireworks.test/chat/completions",
            400,
            "bad request",
            {},
            self._http_error_body(b'{"error":"reasoning_effort unsupported"}'),
        )
        responses = [http_error, self._urlopen_response(success)]

        def fake_urlopen(*args, **kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            client = holdout.InstrumentedProductionFireworksClient(self._fake_fireworks_config())
            self.assertEqual(client.answer(prompt="p", category="math", model="kimi-k2p7-code", task_id="t1"), "remote answer")
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["success"])
        self.assertEqual(client.calls[0]["http_attempt_count"], 2)
        self.assertEqual(client.calls[0]["retry_count"], 1)
        self.assertEqual(client.calls[0]["total_tokens"], 7)

    def test_production_fireworks_records_failed_and_successful_top_level_calls(self) -> None:
        http_error = urllib.error.HTTPError(
            "http://fireworks.test/chat/completions",
            503,
            "unavailable",
            {},
            self._http_error_body(b"temporary outage"),
        )
        success = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        responses = [http_error, self._urlopen_response(success)]

        def fake_urlopen(*args, **kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            client = holdout.InstrumentedProductionFireworksClient(self._fake_fireworks_config())
            with self.assertRaises(Exception):
                client.answer(prompt="p", category="math", model="kimi-a", task_id="t1")
            self.assertEqual(client.answer(prompt="p", category="math", model="minimax-b", task_id="t1"), "ok")
        self.assertEqual(len(client.calls), 2)
        self.assertFalse(client.calls[0]["success"])
        self.assertTrue(client.calls[1]["success"])
        self.assertFalse(client.calls[1]["usage_known"])

    def test_audit_required_before_semantic_readiness(self) -> None:
        self.assertEqual(holdout.readiness_decision({"strict_final_score": 48}, True), "AUDIT_REQUIRED")

    def test_audit_does_not_modify_strict_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            strict = {"strict_final_score": 48}
            strict_path = out / "strict_automated_report.json"
            holdout.atomic_write_json(strict_path, strict)
            strict_hash = holdout.sha256_file(strict_path)
            holdout.atomic_write_json(out / "semantic_audit_queue.json", {"strict_report_hash": strict_hash, "rows": []})
            holdout.atomic_write_json(out / "final_integrity.json", {"overall_ok": True})
            classifications = out / "classifications.json"
            holdout.atomic_write_json(classifications, {"semantic_counts": {"deterministic_solver_wrong_count": 0, "verified_local_objectively_wrong_count": 0, "repair_regression_count": 0}, "classifications": []})
            rc = holdout.audit(Namespace(output_dir=str(out), classifications=str(classifications)))
            self.assertEqual(rc, 0)
            self.assertEqual(holdout.sha256_file(strict_path), strict_hash)
            with self.assertRaises(SystemExit):
                holdout.audit(Namespace(output_dir=str(out), classifications=str(classifications)))

    def test_pre_task_model_failure_does_not_consume_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path, expected_path, manifest_path, artifact = self._synthetic_run_context(tmp)
            with self._patch_run_globals(tasks_path, expected_path, manifest_path):
                with patch.object(holdout, "validate_preflight"), patch("minima.config.load_config", return_value=SimpleNamespace()), patch("minima.local_llm._load_model", return_value=None):
                    with self.assertRaises(SystemExit):
                        holdout.run_holdout(self._run_args(artifact))
            self.assertFalse((artifact / "one_shot_lock.json").exists())

    def test_config_failure_does_not_consume_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path, expected_path, manifest_path, artifact = self._synthetic_run_context(tmp)
            with self._patch_run_globals(tasks_path, expected_path, manifest_path):
                with patch.object(holdout, "validate_preflight"), patch("minima.config.load_config", side_effect=RuntimeError("config")):
                    with self.assertRaises(RuntimeError):
                        holdout.run_holdout(self._run_args(artifact))
            self.assertFalse((artifact / "one_shot_lock.json").exists())

    def test_orchestrator_construction_failure_does_not_consume_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path, expected_path, manifest_path, artifact = self._synthetic_run_context(tmp)
            with self._patch_run_globals(tasks_path, expected_path, manifest_path):
                with patch.object(holdout, "validate_preflight"), patch("minima.config.load_config", return_value=SimpleNamespace()), patch.object(holdout, "InstrumentedProductionFireworksClient", return_value=SimpleNamespace(calls=[])), patch("minima.runtime.LocalFirstOrchestrator", side_effect=RuntimeError("ctor")):
                    with self.assertRaises(RuntimeError):
                        holdout.run_holdout(self._run_args(artifact))
            self.assertFalse((artifact / "one_shot_lock.json").exists())

    def test_failure_during_task_one_marks_lock_failed(self) -> None:
        class BadOrchestrator:
            def __init__(self, client): pass
            def answer_with_trace(self, prompt, task_id=None):
                raise RuntimeError("task boom")

        with tempfile.TemporaryDirectory() as tmp:
            tasks_path, expected_path, manifest_path, artifact = self._synthetic_run_context(tmp)
            with self._patch_run_globals(tasks_path, expected_path, manifest_path):
                with patch.object(holdout, "validate_preflight"), patch("minima.config.load_config", return_value=SimpleNamespace()), patch.object(holdout, "InstrumentedProductionFireworksClient", return_value=SimpleNamespace(calls=[])), patch("minima.local_llm._load_model", return_value=object()), patch("minima.local_llm.model_initialization_count", return_value=1), patch("minima.runtime.LocalFirstOrchestrator", BadOrchestrator):
                    with self.assertRaises(RuntimeError):
                        holdout.run_holdout(self._run_args(artifact))
            self.assertEqual(json.loads((artifact / "one_shot_lock.json").read_text())["status"], "failed")
            self.assertTrue((artifact / "exec" / "results.failed.json").exists())

    def test_complete_three_task_synthetic_run_marks_completed(self) -> None:
        class GoodOrchestrator:
            def __init__(self, client): pass
            def answer_with_trace(self, prompt, task_id=None):
                return HoldoutEvaluatorPreparationTests()._fake_trace(task_id or "", str(task_id)[1:])

        with tempfile.TemporaryDirectory() as tmp:
            tasks_path, expected_path, manifest_path, artifact = self._synthetic_run_context(tmp)
            out = artifact / "exec"
            with self._patch_run_globals(tasks_path, expected_path, manifest_path):
                with patch.object(holdout, "validate_preflight"), patch("minima.config.load_config", return_value=SimpleNamespace()), patch.object(holdout, "InstrumentedProductionFireworksClient", return_value=SimpleNamespace(calls=[])), patch("minima.local_llm._load_model", return_value=object()), patch("minima.local_llm.model_initialization_count", return_value=1), patch("minima.runtime.LocalFirstOrchestrator", GoodOrchestrator):
                    rc = holdout.run_holdout(self._run_args(artifact, out))
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads((artifact / "one_shot_lock.json").read_text())["status"], "completed")
            results = json.loads((out / "results.json").read_text())
            self.assertTrue(all(set(row) == {"task_id", "answer"} for row in results))
            strict = json.loads((out / "strict_automated_report.json").read_text())
            self.assertEqual(strict["automated_status"], "HOLDOUT_EXECUTION_COMPLETE_AUDIT_REQUIRED")

    def test_deadline_failure_marks_started_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path, expected_path, manifest_path, artifact = self._synthetic_run_context(tmp)
            with self._patch_run_globals(tasks_path, expected_path, manifest_path), patch.object(holdout, "RUN_DEADLINE_SECONDS", -1.0):
                with patch.object(holdout, "validate_preflight"), patch("minima.config.load_config", return_value=SimpleNamespace()), patch.object(holdout, "InstrumentedProductionFireworksClient", return_value=SimpleNamespace(calls=[])), patch("minima.local_llm._load_model", return_value=object()), patch("minima.local_llm.model_initialization_count", return_value=1), patch("minima.runtime.LocalFirstOrchestrator", return_value=SimpleNamespace()):
                    rc = holdout.run_holdout(self._run_args(artifact))
            self.assertEqual(rc, 1)
            self.assertFalse((artifact / "one_shot_lock.json").exists())


if __name__ == "__main__":
    unittest.main()
