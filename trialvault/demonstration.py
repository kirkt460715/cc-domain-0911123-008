import tempfile
from pathlib import Path
from .service import Vault
from .validation import canonical
from .releases.exporter import verify_directory


def demonstrate():
    with tempfile.TemporaryDirectory() as directory:
        with Vault(Path(directory) / "vault") as vault:
            schema = {
                "id": "measurements",
                "version": 1,
                "fields": [
                    {"name": "id", "type": "text"},
                    {"name": "pressure", "type": "decimal", "minimum": "0"},
                ],
                "primary_key": ["id"],
            }
            vault.registry.add_schema(schema)
            vault.registry.register(
                "trial", "Synthetic trial measurements", "measurements", "operator"
            )
            content = (
                canonical({"id": "sample-A", "pressure": "12.5"})
                + "\n"
                + canonical({"id": "sample-B", "pressure": "13"})
                + "\n"
            ).encode()
            ingested = vault.ingestion.ingest(
                "request", "trial", 1, content, "jsonl", "internal", ["research"]
            )
            vault.releases.plan(
                "release", [{"dataset": "trial", "version": 1}], "research", "internal"
            )
            vault.releases.approve(
                "release", 1, "independent-reviewer", "Synthetic dataset verified"
            )
            exported = vault.exporter.export(
                "export", "release", Path(directory) / "release"
            )
            verified = verify_directory(Path(directory) / "release")
            assert ingested["rows"] == 2
            assert verified["rows"] == 2
            assert exported["state"] == "complete"
            from .reliability.demo import demonstrate_records

            context = __import__("examples.context", fromlist=["load"]).load()
            life = demonstrate_records(
                context["inspection_records"], Path(directory) / "life-vault"
            )
            return {
                "ingestion": ingested,
                "export": exported,
                "verification": verified,
                "integrity": vault.checks.inspect(),
                "life_experiment": {
                    "status": life["before_revocation"]["status"],
                    "observation_kinds": life["before_revocation"]["observation_kinds"],
                    "selected_model": life["before_revocation"]["selected_model"],
                    "estimates": life["before_revocation"]["estimates"],
                    "bootstrap": life["before_revocation"]["bootstrap"],
                    "export_digest": life["export"]["digest"],
                    "recomputed": life["verification_recomputed"]["recomputed"],
                    "after_revocation": life["after_revocation"],
                    "stale_export_refused": life["stale_export_refused"],
                },
                "likelihood_check": life["likelihood_check"],
                "input_check": life["input"],
            }
