"""Demonstration of a saved life experiment on the synthetic seal-aging data.

Shows interval-censored failures and still-running specimens, whole-batch
bootstrap intervals, the frozen release binding, a verifiable export, and what
happens when the originating release is revoked.
"""

import tempfile
from pathlib import Path

from ..validation import VaultError, canonical
from .spec import celsius_to_kelvin
from .store import verify_life_export


INSPECTION_FIELDS = [
    {"name": "specimen", "type": "text"},
    {"name": "batch", "type": "text"},
    {"name": "temperature_c", "type": "decimal"},
    {"name": "entry_day", "type": "decimal"},
    {"name": "inspection_day", "type": "decimal"},
    {"name": "result", "type": "enum", "choices": ["normal", "failed"]},
]
CONFIG = {
    "use_temperature_c": 25,
    "reference_temperature_c": 25,
    "seed": 7,
    "bootstrap_replicates": 100,
    "validation_fraction": 0.3,
    "survival_points": 31,
}


def demonstrate_likelihood():
    """Independent small-sample likelihood and temperature-unit check."""
    from .analysis import verify_likelihood_terms

    return verify_likelihood_terms()


def demonstrate_records(rows, vault_root):
    """Run the full saved-experiment lifecycle against a temporary vault."""
    from ..service import Vault

    with Vault(Path(vault_root)) as vault:
        vault.registry.add_schema(
            {
                "id": "aging-inspection",
                "version": 1,
                "fields": INSPECTION_FIELDS,
                "primary_key": ["specimen", "inspection_day"],
            }
        )
        vault.registry.register(
            "seal-aging", "Synthetic seal aging inspections", "aging-inspection", "engineer"
        )
        ingested = vault.ingestion.ingest(
            "aging-inspections",
            "seal-aging",
            1,
            ("\n".join(canonical(row) for row in rows) + "\n").encode(),
            "jsonl",
            "internal",
            ["research"],
        )
        vault.releases.plan(
            "aging-release", [{"dataset": "seal-aging", "version": 1}], "research", "internal"
        )
        vault.releases.approve(
            "aging-release", 1, "independent-reviewer", "Inspections cross-checked"
        )
        saved = vault.life.run("seal-life", "seal-aging", 1, "aging-release", CONFIG)
        analysis = saved["analysis"]
        target = Path(vault_root).parent / "life-export"
        exported = vault.life.export("life-export", "seal-life", target)
        verified = verify_life_export(target)
        recomputed = verify_life_export(target, recompute=True)
        before = {
            "status": analysis["status"],
            "restrictions": analysis["restrictions"],
            "observation_kinds": analysis["input"]["observation_kinds"],
            "selected_model": analysis["model_comparison"]["selected"],
            "validation_split": analysis["validation_split"],
            "temperature_range": analysis["temperature_range"],
            "fit": analysis["fit"],
            "estimates": {
                key: analysis["estimates"][key]
                for key in ("eta_days", "B10_days", "B50_days")
            },
            "bootstrap": {
                key: analysis["bootstrap"][key]
                for key in (
                    "attempted",
                    "succeeded",
                    "failed",
                    "failure_rate",
                    "failure_codes",
                    "quantiles",
                )
            },
            "saved_digest": saved["digest"],
            "suitable_for_delivery": saved["suitable_for_delivery"],
        }
        vault.releases.revoke("aging-release", "Synthetic release withdrawn")
        stale = vault.life.get("seal-life")
        after = {
            "suitable_for_delivery": stale["suitable_for_delivery"],
            "reasons": [reason["code"] for reason in stale["suitability_reasons"]],
            "digest_unchanged": stale["digest"] == saved["digest"],
            "bound_version": stale["bound_version"],
        }
        refusal = None
        try:
            vault.life.export("late-export", "seal-life", Path(vault_root).parent / "late")
        except Exception as exc:  # a stale result must not be deliverable
            if isinstance(exc, VaultError):
                refusal = {"code": exc.code, "message": str(exc)}
            else:
                raise
        return {
            "ingestion": ingested,
            "before_revocation": before,
            "export": exported,
            "verification": verified,
            "verification_recomputed": recomputed,
            "after_revocation": after,
            "stale_export_refused": refusal,
            "likelihood_check": demonstrate_likelihood(),
            "input": {
                "synthetic": True,
                "rows": len(rows),
                "use_temperature_c": CONFIG["use_temperature_c"],
                "kelvin_offset_check": {
                    "25_c_kelvin": celsius_to_kelvin(25.0),
                    "100_c_kelvin": celsius_to_kelvin(100.0),
                    "absolute_zero_c_kelvin": celsius_to_kelvin(-273.15),
                    "definition": "T_K = T_C + 273.15 exactly",
                },
            },
        }


def demonstrate():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import examples.context as context

    data = context.load()
    with tempfile.TemporaryDirectory() as directory:
        result = demonstrate_records(
            data["inspection_records"], Path(directory) / "vault"
        )
    result["input"] = {
        "synthetic": data.get("synthetic", False),
        "rows": len(data["inspection_records"]),
        "summary_records": len(data["records"]),
        "use_temperature_c": data["use_temperature_c"],
        "kelvin_offset_check": {
            "25_c_kelvin": celsius_to_kelvin(25.0),
            "100_c_kelvin": celsius_to_kelvin(100.0),
            "absolute_zero_c_kelvin": celsius_to_kelvin(-273.15),
            "definition": "T_K = T_C + 273.15 exactly",
        },
    }
    result["likelihood_check"] = demonstrate_likelihood()
    return result
