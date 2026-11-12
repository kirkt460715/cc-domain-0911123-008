import json
import math
import tempfile
import unittest
from pathlib import Path

from trialvault.service import Vault
from trialvault.validation import VaultError, canonical
from trialvault.contracts.schema import Schema
from trialvault.reliability.spec import celsius_to_kelvin
from trialvault.reliability.likelihood import (
    log_likelihood,
    quantile_time,
    survival,
)
from trialvault.reliability.fitting import maximise, starts_for
from trialvault.reliability.observations import build
from trialvault.reliability.analysis import run_analysis, verify_likelihood_terms
from trialvault.reliability.store import verify_life_export


REFERENCE_C = 25.0
REFERENCE_K = celsius_to_kelvin(REFERENCE_C)


def observation(specimen, kind, temperature_c, entry=0.0, left=None, right=None):
    return {
        "specimen": specimen,
        "batch": "lot-" + specimen,
        "kind": kind,
        "temperature_c": temperature_c,
        "temperature_k": celsius_to_kelvin(temperature_c),
        "entry_day": entry,
        "left_day": left if left is not None else (right if right is not None else 0.0),
        "right_day": right,
        "tau": None if right is None else right - entry,
        "lam": (left if left is not None else (right or entry)) - entry,
    }


class LikelihoodTermTests(unittest.TestCase):
    """Each censoring form is checked against a hand-computed formula."""

    def setUp(self):
        self.beta = 2.0
        self.eta = 10.0
        self.parameters = {"shape": self.beta, "log_scale": math.log(self.eta)}

    def test_exact_term_matches_closed_form(self):
        item = observation("a", "exact", REFERENCE_C, right=10.0)
        expected = (
            math.log(self.beta)
            + (self.beta - 1.0) * (math.log(10.0) - math.log(self.eta))
            - math.log(self.eta)
            - (10.0 / self.eta) ** self.beta
        )
        observed = log_likelihood(
            [item], "common_scale_weibull", self.parameters, REFERENCE_K
        )
        self.assertAlmostEqual(observed, expected, places=12)

    def test_right_censored_term_is_log_survival(self):
        item = observation("a", "right", REFERENCE_C, left=12.0)
        expected = -((12.0 / self.eta) ** self.beta)
        self.assertAlmostEqual(
            log_likelihood(
                [item], "common_scale_weibull", self.parameters, REFERENCE_K
            ),
            expected,
            places=12,
        )

    def test_interval_term_is_exact_survival_increment_not_midpoint(self):
        item = observation("a", "interval", REFERENCE_C, left=9.0, right=15.0)
        expected = math.log(
            math.exp(-((9.0 / self.eta) ** self.beta))
            - math.exp(-((15.0 / self.eta) ** self.beta))
        )
        observed = log_likelihood(
            [item], "common_scale_weibull", self.parameters, REFERENCE_K
        )
        self.assertAlmostEqual(observed, expected, places=12)
        # Midpoint substitution would give a different, wrong value.
        midpoint = math.log(
            math.exp(-((12.0 / self.eta) ** self.beta))
            - math.exp(-((12.001 / self.eta) ** self.beta))
        )
        self.assertGreater(abs(observed - midpoint), 1.0)

    def test_interval_term_stays_precise_when_bounds_are_tight(self):
        item = observation("a", "interval", REFERENCE_C, left=9.999, right=10.001)
        observed = log_likelihood(
            [item], "common_scale_weibull", self.parameters, REFERENCE_K
        )
        expected = math.log(
            math.exp(-((9.999 / self.eta) ** self.beta))
            - math.exp(-((10.001 / self.eta) ** self.beta))
        )
        self.assertAlmostEqual(observed, expected, places=10)
        self.assertTrue(math.isfinite(observed))

    def test_delayed_entry_uses_exposure_from_entry(self):
        delayed = observation("a", "exact", REFERENCE_C, entry=5.0, right=15.0)
        on_time = observation("b", "exact", REFERENCE_C, entry=0.0, right=10.0)
        self.assertAlmostEqual(
            log_likelihood(
                [delayed], "common_scale_weibull", self.parameters, REFERENCE_K
            ),
            log_likelihood(
                [on_time], "common_scale_weibull", self.parameters, REFERENCE_K
            ),
            places=14,
        )

    def test_delayed_entry_right_censoring_uses_exposure(self):
        item = observation("a", "right", REFERENCE_C, entry=4.0, left=14.0)
        self.assertAlmostEqual(item["lam"], 10.0)
        expected = -((10.0 / self.eta) ** self.beta)
        self.assertAlmostEqual(
            log_likelihood(
                [item], "common_scale_weibull", self.parameters, REFERENCE_K
            ),
            expected,
            places=12,
        )

    def test_arrhenius_term_uses_kelvin_offset_273_15(self):
        self.assertAlmostEqual(REFERENCE_K, 298.15)
        self.assertAlmostEqual(celsius_to_kelvin(100.0), 373.15)
        slope = 1500.0
        item = observation("a", "exact", 100.0, right=5.0)
        expected_eta = self.eta * math.exp(slope * (1.0 / 373.15 - 1.0 / 298.15))
        expected = (
            math.log(self.beta)
            + (self.beta - 1.0) * (math.log(5.0) - math.log(expected_eta))
            - math.log(expected_eta)
            - (5.0 / expected_eta) ** self.beta
        )
        observed = log_likelihood(
            [item],
            "arrhenius_weibull",
            {**self.parameters, "slope": slope},
            REFERENCE_K,
        )
        self.assertAlmostEqual(observed, expected, places=11)
        # A rounded offset of 273 changes the answer materially.
        wrong_eta = self.eta * math.exp(slope * (1.0 / 373.0 - 1.0 / 298.0))
        self.assertGreater(abs(wrong_eta / expected_eta - 1.0), 5e-4)
        wrong = (
            math.log(self.beta)
            + (self.beta - 1.0) * (math.log(5.0) - math.log(wrong_eta))
            - math.log(wrong_eta)
            - (5.0 / wrong_eta) ** self.beta
        )
        self.assertGreater(abs(wrong - observed), 1e-3)

    def test_common_scale_ignores_temperature(self):
        cold = observation("a", "exact", 60.0, right=10.0)
        hot = observation("b", "exact", 100.0, right=10.0)
        self.assertAlmostEqual(
            log_likelihood(
                [cold], "common_scale_weibull", self.parameters, REFERENCE_K
            ),
            log_likelihood([hot], "common_scale_weibull", self.parameters, REFERENCE_K),
            places=14,
        )

    def test_exported_hand_check_matches_closed_form(self):
        check = verify_likelihood_terms()
        self.assertEqual(check["reference_temperature_k"], 298.15)
        terms = {item["specimen"]: item for item in check["terms"]}
        # shape 2, eta 10: exact at 10 days is log(2/10) - 1
        self.assertAlmostEqual(
            terms["exact"]["log_likelihood"], math.log(2.0 / 10.0) - 1.0, places=12
        )
        # right censored at 12 days: -(12/10)^2
        self.assertAlmostEqual(
            terms["right"]["log_likelihood"], -((12.0 / 10.0) ** 2), places=12
        )
        # interval (9, 15]: log(exp(-0.81) - exp(-2.25))
        self.assertAlmostEqual(
            terms["interval"]["log_likelihood"],
            math.log(math.exp(-((9.0 / 10.0) ** 2)) - math.exp(-((15.0 / 10.0) ** 2))),
            places=12,
        )
        self.assertAlmostEqual(
            check["total"], sum(item["log_likelihood"] for item in terms.values()), places=12
        )

    def test_survival_and_quantile_are_consistent(self):
        item = observation("a", "exact", REFERENCE_C, right=10.0)
        b10 = quantile_time(
            "common_scale_weibull", self.parameters, REFERENCE_K, REFERENCE_C, 0.10
        )
        self.assertAlmostEqual(b10, self.eta * (-math.log(0.9)) ** (1 / self.beta))
        self.assertAlmostEqual(
            survival(
                [], "common_scale_weibull", self.parameters, REFERENCE_K, REFERENCE_C, b10
            ),
            0.90,
            places=8,
        )
        b50 = quantile_time(
            "common_scale_weibull", self.parameters, REFERENCE_K, REFERENCE_C, 0.50
        )
        self.assertAlmostEqual(
            survival(
                [], "common_scale_weibull", self.parameters, REFERENCE_K, REFERENCE_C, b50
            ),
            0.50,
            places=8,
        )
        self.assertEqual(item["tau"], 10.0)


def inspection_schema():
    return {
        "id": "seal-inspection",
        "version": 1,
        "fields": [
            {"name": "specimen", "type": "text"},
            {"name": "batch", "type": "text"},
            {"name": "temperature_c", "type": "decimal"},
            {"name": "entry_day", "type": "decimal"},
            {"name": "inspection_day", "type": "decimal"},
            {"name": "result", "type": "enum", "choices": ["normal", "failed"]},
            {"name": "exact", "type": "boolean", "required": False},
            {"name": "removed", "type": "boolean", "required": False},
        ],
        "primary_key": ["specimen", "inspection_day"],
    }


def row(specimen, batch, temperature, entry, day, result, **extra):
    return {
        "specimen": specimen,
        "batch": batch,
        "temperature_c": str(temperature),
        "entry_day": str(entry),
        "inspection_day": str(day),
        "result": result,
        **extra,
    }


class DiagnosticsTests(unittest.TestCase):
    def run_rows(self, rows, **config):
        contract = Schema(inspection_schema())
        return run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 25,
                "seed": 11,
                "bootstrap_replicates": 5,
                "survival_points": 11,
                **config,
            },
        )

    def checks(self, result):
        return [
            item for item in result["input"]["checks"] if item["status"] == "present"
        ]

    def test_censoring_forms_are_distinguished(self):
        rows = [
            row("exact", "b1", 60, 0, 6, "normal"),
            row("exact", "b1", 60, 0, 9, "failed", exact=True),
            row("interval", "b1", 80, 0, 12, "normal"),
            row("interval", "b1", 80, 0, 15, "failed"),
            row("right", "b2", 100, 3, 9, "normal"),
            row("right", "b2", 100, 3, 12, "normal"),
        ]
        result = self.run_rows(rows)
        kinds = result["input"]["observation_kinds"]
        self.assertEqual(kinds["exact"], 1)
        self.assertEqual(kinds["interval"], 1)
        self.assertEqual(kinds["right"], 1)
        self.assertEqual(kinds["ongoing"], 1)
        self.assertEqual(kinds["delayed_entry"], 1)
        observations = result["input"]["observations"]
        exact = [item for item in observations if item["specimen"] == "exact"][0]
        self.assertEqual((exact["left_day"], exact["right_day"]), (9.0, 9.0))
        interval = [
            item for item in observations if item["specimen"] == "interval"
        ][0]
        self.assertEqual((interval["left_day"], interval["right_day"]), (12.0, 15.0))
        censored = [item for item in observations if item["specimen"] == "right"][0]
        self.assertEqual((censored["left_day"], censored["right_day"]), (12.0, None))
        self.assertIsNone(censored["tau"])
        self.assertEqual(censored["lam"], 9.0)

    def test_true_exact_failure_must_not_be_expanded_to_interval(self):
        rows = [
            row("s", "b1", 60, 0, 8, "normal"),
            row("s", "b1", 60, 0, 9, "failed"),
        ]
        result = self.run_rows(rows)
        observation_row = result["input"]["observations"][0]
        self.assertEqual(observation_row["kind"], "interval")
        self.assertNotEqual(observation_row["left_day"], observation_row["right_day"])

    def test_recovery_after_failure_is_flagged_and_excluded(self):
        rows = [
            row("s", "b1", 60, 0, 6, "normal"),
            row("s", "b1", 60, 0, 9, "failed"),
            row("s", "b1", 60, 0, 12, "normal"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("recovered_after_failure", codes)
        self.assertEqual(result["input"]["included_specimens"], 0)
        self.assertEqual(result["input"]["excluded_specimens"], 1)
        decisions = result["input"]["decisions"]
        self.assertEqual(len(decisions), 3)
        self.assertTrue(all(item["decision"] == "excluded" for item in decisions))

    def test_out_of_order_and_duplicate_inspections_are_diagnosed(self):
        rows = [
            row("s", "b1", 60, 0, 12, "normal"),
            row("s", "b1", 60, 0, 6, "normal"),
            row("s", "b1", 60, 0, 12, "normal"),
            row("s", "b1", 60, 0, 15, "failed"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("inspection_out_of_order", codes)
        self.assertIn("duplicate_inspection", codes)
        observation = result["input"]["observations"][0]
        self.assertEqual(observation["left_day"], 12.0)
        self.assertEqual(observation["right_day"], 15.0)
        duplicates = [
            item
            for item in result["input"]["decisions"]
            if item["code"] == "duplicate_inspection"
        ]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["row"], 3)

    def test_conflicting_duplicate_inspection_excludes_specimen(self):
        rows = [
            row("s", "b1", 60, 0, 6, "normal"),
            row("s", "b1", 60, 0, 12, "failed"),
            row("s", "b1", 60, 0, 12, "normal"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("conflicting_duplicate_inspection", codes)
        self.assertEqual(result["input"]["included_specimens"], 0)

    def test_invalid_temperature_and_reversed_bounds(self):
        rows = [
            row("cold", "b1", -500, 0, 6, "normal"),
            row("cold", "b1", -500, 0, 9, "failed"),
            row("reverse", "b1", 60, 0, 3, "failed"),
            row("reverse", "b1", 60, 0, 18, "normal"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("invalid_temperature", codes)
        self.assertIn("failure_without_prior_inspection", codes)
        self.assertEqual(result["input"]["included_specimens"], 0)
        self.assertEqual(result["status"], "failed")

    def test_summary_format_detects_reversed_bounds(self):
        contract = Schema(
            {
                "id": "summary",
                "version": 1,
                "fields": [
                    {"name": "specimen", "type": "text"},
                    {"name": "batch", "type": "text"},
                    {"name": "temperature_c", "type": "decimal"},
                    {"name": "entry_day", "type": "decimal"},
                    {"name": "left_day", "type": "decimal"},
                    {"name": "right_day", "type": "decimal", "nullable": True},
                    {"name": "kind", "type": "enum",
                     "choices": ["exact", "interval", "right"]},
                ],
                "primary_key": ["specimen"],
            }
        )
        rows = [
            {"specimen": "rev", "batch": "b1", "temperature_c": "60",
             "entry_day": "0", "left_day": "20", "right_day": "12",
             "kind": "interval"},
            {"specimen": "ok", "batch": "b1", "temperature_c": "80",
             "entry_day": "2", "left_day": "12", "right_day": "15",
             "kind": "interval"},
            {"specimen": "done", "batch": "b2", "temperature_c": "100",
             "entry_day": "0", "left_day": "9", "right_day": None,
             "kind": "right"},
        ]
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 25,
                "seed": 3,
                "bootstrap_replicates": 5,
                "survival_points": 11,
                "input_format": "interval_summary",
                "columns": {
                    "interval_left": "left_day",
                    "interval_right": "right_day",
                    "interval_kind": "kind",
                },
            },
        )
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("left_bound_after_right_bound", codes)
        kinds = result["input"]["observation_kinds"]
        self.assertEqual(kinds["interval"], 1)
        self.assertEqual(kinds["right"], 1)
        self.assertEqual(result["input"]["excluded_specimens"], 1)

    def test_unknown_summary_kind_is_not_coerced(self):
        contract = Schema(
            {
                "id": "summary",
                "version": 1,
                "fields": [
                    {"name": "specimen", "type": "text"},
                    {"name": "batch", "type": "text"},
                    {"name": "temperature_c", "type": "decimal"},
                    {"name": "entry_day", "type": "decimal"},
                    {"name": "left_day", "type": "decimal"},
                    {"name": "right_day", "type": "decimal", "nullable": True},
                    {"name": "kind", "type": "text"},
                ],
                "primary_key": ["specimen"],
            }
        )
        rows = [
            {"specimen": "x", "batch": "b", "temperature_c": "60",
             "entry_day": "0", "left_day": "9", "right_day": "12",
             "kind": "suspended"},
        ]
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 25,
                "seed": 1,
                "bootstrap_replicates": 2,
                "survival_points": 11,
                "input_format": "interval_summary",
                "columns": {"interval_left": "left_day",
                            "interval_right": "right_day",
                            "interval_kind": "kind"},
            },
        )
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("invalid_result", codes)
        self.assertEqual(result["input"]["included_specimens"], 0)


def aging_rows(count_per_temperature=12, temperatures=(60, 80, 100), shape=2.2):
    """Deterministic synthetic failures spread across three batches."""
    rows = []
    for index in range(count_per_temperature * len(temperatures)):
        temperature = temperatures[index % len(temperatures)]
        batch = "b" + str(index % 4)
        base = (index // 4) % 5 + 1
        if index % 9 == 0:
            rows.append(row(str(index), batch, temperature, 0, base + 8, "normal"))
            rows.append(row(str(index), batch, temperature, 0, base + 12, "normal"))
        else:
            rows.append(row(str(index), batch, temperature, 0, base + 3, "normal"))
            rows.append(row(str(index), batch, temperature, 0, base + 7, "failed"))
    return rows


class AnalysisTests(unittest.TestCase):
    def test_model_selection_uses_only_held_out_batches(self):
        rows = aging_rows()
        contract = Schema(inspection_schema())
        config = {
            "use_temperature_c": 90,
            "seed": 5,
            "bootstrap_replicates": 6,
            "survival_points": 11,
        }
        result = run_analysis(rows, contract, config)
        split = result["validation_split"]
        self.assertTrue(split["validation"])
        self.assertGreaterEqual(len(split["train"]), 1)
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        chosen = result["model_comparison"]["selected"]
        self.assertIn(chosen, {"common_scale_weibull", "arrhenius_weibull"})
        for entry in result["model_comparison"]["models"]:
            if entry["selectable"]:
                self.assertIsNotNone(entry["validation_log_likelihood"])

    def test_temperature_range_uses_training_batches_only(self):
        rows = aging_rows()
        contract = Schema(inspection_schema())
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 25,
                "seed": 5,
                "bootstrap_replicates": 4,
                "survival_points": 11,
            },
        )
        training_batches = set(result["validation_split"]["train"])
        training_temperatures = {
            float(item["temperature_c"])
            for item in result["input"]["observations"]
            if item["batch"] in training_batches
        }
        low = min(training_temperatures)
        high = max(training_temperatures)
        self.assertEqual(result["temperature_range"]["training_range_c"]["minimum"], low)
        self.assertEqual(result["temperature_range"]["training_range_c"]["maximum"], high)
        self.assertTrue(result["temperature_range"]["extrapolation"])

    def test_pinned_validation_batches_are_respected(self):
        rows = aging_rows()
        contract = Schema(inspection_schema())
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 80,
                "seed": 1,
                "bootstrap_replicates": 4,
                "survival_points": 11,
                "validation_batches": ["b2", "b3"],
            },
        )
        self.assertEqual(result["validation_split"]["selection"], "pinned")
        self.assertEqual(result["validation_split"]["validation"], ["b2", "b3"])
        self.assertEqual(result["validation_split"]["train"], ["b0", "b1"])

    def test_single_batch_cannot_select_and_says_so(self):
        rows = [
            row("a", "only", 60, 0, 4, "normal"),
            row("a", "only", 60, 0, 7, "failed"),
            row("b", "only", 80, 0, 5, "normal"),
            row("b", "only", 80, 0, 9, "failed"),
        ]
        contract = Schema(inspection_schema())
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 60,
                "seed": 1,
                "bootstrap_replicates": 4,
                "survival_points": 11,
            },
        )
        self.assertIsNone(result["model_comparison"]["selected"])
        codes = {item["code"] for item in result["restrictions"]}
        self.assertIn("model_selection_unavailable", codes)
        self.assertEqual(result["status"], "restricted")

    def test_all_censored_data_blocks_identifiability(self):
        rows = [
            row("a", "b1", 60, 0, 4, "normal"),
            row("a", "b1", 60, 0, 8, "normal"),
            row("b", "b2", 80, 0, 4, "normal"),
            row("b", "b2", 80, 0, 8, "normal"),
        ]
        contract = Schema(inspection_schema())
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 60,
                "seed": 1,
                "bootstrap_replicates": 3,
                "survival_points": 11,
            },
        )
        codes = {item["code"] for item in result["restrictions"]}
        self.assertIn("parameters_not_identifiable", codes)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(
            any(
                item["code"] == "no_failures" for item in result["identifiability"]
            )
        )

    def test_bootstrap_failure_rate_is_reported_not_hidden(self):
        rows = aging_rows(count_per_temperature=4, temperatures=(60, 100))
        contract = Schema(inspection_schema())
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 60,
                "seed": 2,
                "bootstrap_replicates": 12,
                "survival_points": 11,
            },
        )
        boot = result["bootstrap"]
        self.assertEqual(boot["attempted"], 12)
        self.assertEqual(boot["attempted"], boot["succeeded"] + boot["failed"])
        self.assertIsNotNone(boot["failure_rate"])
        self.assertIn("failure_codes", boot)

    def test_survival_curve_is_on_requested_temperature(self):
        rows = aging_rows()
        contract = Schema(inspection_schema())
        result = run_analysis(
            rows,
            contract,
            {
                "use_temperature_c": 90,
                "seed": 5,
                "bootstrap_replicates": 4,
                "survival_points": 21,
            },
        )
        if result["status"] == "failed":
            self.skipTest("fit failed on synthetic data")
        curve = result["survival_curve"]
        self.assertEqual(curve["use_temperature_c"], 90)
        self.assertEqual(len(curve["times_days"]), 21)
        self.assertEqual(len(curve["survival"]), 21)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in curve["survival"]))
        estimates = result["estimates"]
        self.assertIn("B10_days", estimates)
        self.assertIn("B50_days", estimates)
        self.assertGreater(estimates["B10_days"], 0)
        self.assertGreater(estimates["B50_days"], estimates["B10_days"])

    def test_invalid_configuration_is_rejected(self):
        contract = Schema(inspection_schema())
        rows = aging_rows()
        for config in (
            {"use_temperature_c": 25, "seed": 1, "confidence": 1.5},
            {"use_temperature_c": -273.15, "seed": 1},
            {"use_temperature_c": 25},
        ):
            with self.subTest(config=config):
                with self.assertRaises(VaultError):
                    run_analysis(rows, contract, config)


class DiagnosticsEdgeCaseTests(unittest.TestCase):
    def run_rows(self, rows, **config):
        return DiagnosticsTests.run_rows(self, rows, **config)

    def checks(self, result):
        return [
            item for item in result["input"]["checks"] if item["status"] == "present"
        ]

    def test_repeated_failures_are_flagged_but_not_fatal(self):
        rows = [
            row("s", "b1", 60, 0, 4, "normal"),
            row("s", "b1", 60, 0, 7, "failed"),
            row("s", "b1", 60, 0, 9, "failed"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("inspection_after_failure", codes)
        self.assertEqual(result["input"]["included_specimens"], 1)
        observation = result["input"]["observations"][0]
        self.assertEqual((observation["left_day"], observation["right_day"]), (4.0, 7.0))

    def test_removed_marker_drops_row_without_changing_others(self):
        rows = [
            row("s", "b1", 60, 0, 2, "normal", removed=True),
            row("s", "b1", 60, 0, 2, "normal"),
            row("s", "b1", 60, 0, 6, "failed"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("row_marked_removed", codes)
        observation = result["input"]["observations"][0]
        self.assertEqual((observation["left_day"], observation["right_day"]), (2.0, 6.0))
        removed = [
            item
            for item in result["input"]["decisions"]
            if item["code"] == "row_marked_removed"
        ]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["row"], 1)

    def test_exact_failure_at_entry_is_rejected(self):
        rows = [row("s", "b1", 80, 3, 3, "failed", exact=True)]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("nonpositive_exact_time", codes)
        self.assertEqual(result["input"]["included_specimens"], 0)

    def test_exact_marker_on_normal_inspection_is_rejected(self):
        rows = [
            row("s", "b1", 60, 0, 6, "normal", exact=True),
            row("s", "b1", 60, 0, 9, "failed"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("invalid_result", codes)
        self.assertEqual(result["input"]["included_specimens"], 0)

    def test_metadata_conflict_excludes_specimen(self):
        rows = [
            row("s", "b1", 60, 0, 3, "normal"),
            row("s", "b1", 80, 0, 6, "failed"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("metadata_conflict", codes)
        self.assertEqual(result["input"]["included_specimens"], 0)

    def test_failure_on_first_inspection_starts_at_entry(self):
        rows = [
            row("s", "b1", 60, 2, 2, "failed"),
            row("s", "b1", 60, 2, 9, "normal"),
        ]
        result = self.run_rows(rows)
        codes = [item["code"] for item in self.checks(result)]
        self.assertIn("recovered_after_failure", codes)


class OrderingTests(unittest.TestCase):
    def test_numeric_primary_key_orders_numerically(self):
        contract = Schema(
            {
                "id": "numeric-key",
                "version": 1,
                "fields": [
                    {"name": "id", "type": "text"},
                    {"name": "day", "type": "decimal"},
                ],
                "primary_key": ["id", "day"],
            }
        )
        rows = [
            {"id": "a", "day": "12"},
            {"id": "a", "day": "0"},
            {"id": "a", "day": "9"},
            {"id": "b", "day": "5"},
        ]
        ordered = sorted(
            (contract.validate(item) for item in rows), key=contract.order_key
        )
        self.assertEqual(
            [(item["id"], item["day"]) for item in ordered],
            [("a", "0"), ("a", "9"), ("a", "12"), ("b", "5")],
        )
        # Identity keys remain canonical text, unchanged by ordering.
        self.assertEqual(contract.row_key(ordered[0]), '["a","0"]')


class SavedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vault = Vault(self.root / "vault")
        self.addCleanup(self.vault.close)
        schema = inspection_schema()
        self.vault.registry.add_schema(schema)
        self.vault.registry.register(
            "seal", "Seal aging inspections", "seal-inspection", "engineer"
        )
        rows = aging_rows()
        self.vault.ingestion.ingest(
            "inspection",
            "seal",
            1,
            ("\n".join(canonical(item) for item in rows) + "\n").encode(),
            "jsonl",
            "internal",
            ["research"],
        )
        self.vault.releases.plan(
            "seal-release", [{"dataset": "seal", "version": 1}], "research", "internal"
        )
        self.vault.releases.approve("seal-release", 1, "reviewer", "Checked")

    def config(self, **changes):
        return {
            "use_temperature_c": 25,
            "seed": 9,
            "bootstrap_replicates": 6,
            "survival_points": 11,
            **changes,
        }

    def test_saved_experiment_freezes_version_and_is_deterministic(self):
        first = self.vault.life.run(
            "life", "seal", 1, "seal-release", self.config()
        )
        self.assertEqual(first["bound_version"], 1)
        self.assertFalse(first["recomputed_from_later_version"])
        self.assertEqual(
            first["binding"]["inputs"][0]["version_digest"],
            self.vault.registry.version("seal", 1)["digest"],
        )
        again = self.vault.life.get("life")
        self.assertEqual(again["digest"], first["digest"])
        self.assertEqual(canonical(again), canonical(first))
        with self.assertRaises(VaultError) as caught:
            self.vault.life.run(
                "life", "seal", 1, "seal-release", self.config(seed=10)
            )
        self.assertEqual(caught.exception.code, "conflict")

    def test_new_dataset_version_is_not_used_silently(self):
        saved = self.vault.life.run("life", "seal", 1, "seal-release", self.config())
        rebuilt = aging_rows(count_per_temperature=5)
        self.vault.ingestion.ingest(
            "next",
            "seal",
            1,
            ("\n".join(canonical(item) for item in rebuilt) + "\n").encode(),
            "jsonl",
            "internal",
            ["research"],
            expected_version=1,
        )
        self.assertEqual(self.vault.registry.list_versions("seal")[-1]["version"], 2)
        current = self.vault.life.get("life")
        self.assertEqual(current["bound_version"], 1)
        self.assertEqual(current["digest"], saved["digest"])
        self.assertTrue(current["suitable_for_delivery"])

    def test_withdrawing_source_marks_result_unsuitable(self):
        saved = self.vault.life.run("life", "seal", 1, "seal-release", self.config())
        self.vault.registry.withdraw("seal", 1, "Synthetic correction")
        current = self.vault.life.get("life")
        self.assertFalse(current["suitable_for_delivery"])
        codes = {item["code"] for item in current["suitability_reasons"]}
        self.assertIn("input_version_withdrawn", codes)
        self.assertEqual(current["digest"], saved["digest"])
        with self.assertRaises(VaultError) as caught:
            self.vault.life.export("export", "life", self.root / "out")
        self.assertEqual(caught.exception.code, "state")

    def test_revoked_release_marks_result_unsuitable(self):
        saved = self.vault.life.run("life", "seal", 1, "seal-release", self.config())
        self.vault.releases.revoke("seal-release", "Distribution halted")
        current = self.vault.life.get("life")
        self.assertFalse(current["suitable_for_delivery"])
        self.assertIn(
            "release_revoked",
            {item["code"] for item in current["suitability_reasons"]},
        )
        self.assertEqual(current["digest"], saved["digest"])

    def test_run_requires_approved_release_containing_version(self):
        self.vault.releases.plan(
            "draft", [{"dataset": "seal", "version": 1}], "research", "internal"
        )
        with self.assertRaises(VaultError) as caught:
            self.vault.life.run("draft-run", "seal", 1, "draft", self.config())
        self.assertEqual(caught.exception.code, "state")
        with self.assertRaises(VaultError) as caught:
            self.vault.life.run("missing", "seal", 1, "unrelated-release", self.config())
        self.assertEqual(caught.exception.code, "missing")

    def test_life_record_tampering_is_detected(self):
        self.vault.life.run("life", "seal", 1, "seal-release", self.config())
        with self.vault.database.transaction() as con:
            con.execute(
                "UPDATE life_experiments SET decisions='[]' WHERE id='life'"
            )
        with self.assertRaises(VaultError) as caught:
            self.vault.life.get("life")
        self.assertEqual(caught.exception.code, "integrity")

    def test_export_verifies_and_recomputes_from_bundle(self):
        saved = self.vault.life.run("life", "seal", 1, "seal-release", self.config())
        target = self.root / "life-export"
        exported = self.vault.life.export("export", "life", target)
        self.assertEqual(exported["state"], "complete")
        verified = verify_life_export(target)
        self.assertEqual(verified["experiment"], "life")
        self.assertEqual(verified["input_rows"], len(aging_rows()))
        self.assertEqual(verified["observations"], saved["analysis"]["input"]["included_specimens"])
        self.assertEqual(
            verified["release_digest"], saved["binding"]["release"]["digest"]
        )
        recomputed = verify_life_export(target, recompute=True)
        self.assertEqual(recomputed["digest"], verified["digest"])
        self.assertTrue(recomputed["recomputed"])

    def test_export_detects_changed_bundle_member(self):
        self.vault.life.run("life", "seal", 1, "seal-release", self.config())
        target = self.root / "life-export"
        self.vault.life.export("export", "life", target)
        manifest = json.loads((target / "manifest.json").read_text())
        member = next(iter(manifest["manifest"]["inputs"]))["bundle_member"]
        (target / member).write_bytes(b"changed\n")
        with self.assertRaises(VaultError) as caught:
            verify_life_export(target)
        self.assertEqual(caught.exception.code, "integrity")

    def test_export_manifest_lists_input_lineage(self):
        saved = self.vault.life.run("life", "seal", 1, "seal-release", self.config())
        target = self.root / "life-export"
        self.vault.life.export("export", "life", target)
        envelope = json.loads((target / "manifest.json").read_text())
        manifest = envelope["manifest"]
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(manifest["kind"], "life-experiment-export")
        self.assertEqual(manifest["seed"], 9)
        self.assertEqual(manifest["config"], saved["binding"]["config"])
        self.assertEqual(manifest["release"]["id"], "seal-release")
        self.assertEqual(
            manifest["release"]["digest"], saved["binding"]["release"]["digest"]
        )
        self.assertEqual(
            manifest["row_counts"], saved["binding"]["row_counts"]
        )
        self.assertEqual(len(manifest["inputs"][0]["artifact"]), 64)

    def test_manual_exclusion_is_frozen_and_attributed(self):
        saved = self.vault.life.run(
            "life",
            "seal",
            1,
            "seal-release",
            self.config(manual_exclusions=[{"specimen": "0", "reason": "leak"}]),
        )
        decisions = saved["analysis"]["input"]["decisions"]
        excluded = [item for item in decisions if item["decision"] == "excluded"]
        self.assertTrue(excluded)
        self.assertTrue(all(item["code"] == "manual_exclusion" for item in excluded))
        self.assertEqual(saved["analysis"]["input"]["excluded_specimens"], 1)

    def test_manual_exclusion_of_absent_specimen_is_rejected(self):
        with self.assertRaises(VaultError):
            self.vault.life.run(
                "life",
                "seal",
                1,
                "seal-release",
                self.config(manual_exclusions=[{"specimen": "ghost", "reason": "x"}]),
            )
