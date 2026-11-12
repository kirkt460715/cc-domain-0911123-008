"""Saved life experiments bound to an approved release version.

A saved experiment freezes:

* the dataset version and its payload/artifact digests,
* the approved release that contains that version, including its reviewer and
  release digest,
* the analysis configuration and the random seed,
* every row-level inclusion decision and the resulting censoring observations,
* the complete analysis payload and its content digest.

Retrieval never re-runs the analysis and never silently picks a newer dataset
version.  When the release is revoked, the bound version withdrawn, or any
frozen input changed, the stored result is still returned but marked
``suitable_for_delivery: false`` with the reasons.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ..validation import canonical, digest, name, parse_json, reject, integer
from ..contracts.schema import Schema
from .analysis import ANALYSIS_VERSION, run_analysis
from .errors import Failure
from .spec import parse_config


def _public_config(config):
    return {
        key: value for key, value in config.items() if not str(key).startswith("_")
    }


class LifeExperiments:
    def __init__(self, database, registry, ingestion, releases, artifacts):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion
        self.releases = releases
        self.artifacts = artifacts
        with database.transaction() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS life_experiments(
                   id TEXT PRIMARY KEY,
                   request TEXT NOT NULL,
                   binding TEXT NOT NULL,
                   analysis TEXT NOT NULL,
                   decisions TEXT NOT NULL,
                   observations TEXT NOT NULL,
                   digest TEXT NOT NULL)"""
            )

    # ------------------------------------------------------------------ run
    def run(self, experiment_id, dataset, version, release, config):
        name(experiment_id)
        name(dataset)
        name(release)
        integer(version, 1)
        try:
            parsed = parse_config(config)
        except Failure as exc:
            reject("config", str(exc), error=exc.as_dict())
        record = self.registry.version(dataset, version)
        if record["state"] != "active":
            reject("state", "life experiment requires an active dataset version")
        release_record = self.releases.get(release)
        if release_record["state"] != "approved":
            reject("state", "life experiment requires an approved release")
        eligibility = self.releases.eligibility(release)
        if not eligibility["eligible"]:
            reject(
                "lineage",
                "life experiment release is no longer eligible",
                problems=eligibility["problems"],
            )
        manifest = release_record["manifest"]
        included = {
            (item["dataset"], item["version"]) for item in manifest["versions"]
        }
        if (dataset, version) not in included:
            reject("input", "release does not contain the bound dataset version")
        approval = self.db.one(
            "SELECT reviewer,manifest_digest,reason FROM approvals WHERE release=?",
            (release,),
        )
        if approval is None or approval["manifest_digest"] != release_record["digest"]:
            reject("approval", "release approval does not match the frozen manifest")
        schema = self.registry.schema(
            self.registry.dataset(dataset)["schema_id"], record["schema_version"]
        )
        rows = self.ingestion.rows(dataset, version)

        request = {
            "dataset": dataset,
            "version": version,
            "release": release,
            "config": _public_config(parsed),
            "analysis_version": ANALYSIS_VERSION,
        }
        with self.db.transaction() as con:
            old = self.db.one("SELECT * FROM life_experiments WHERE id=?", (experiment_id,))
            if old:
                if old["request"] != canonical(request):
                    reject(
                        "conflict",
                        "life experiment id is already bound to a different request",
                    )
                return self._record(experiment_id)
            try:
                analysis = run_analysis(rows, schema, parsed)
            except Failure as exc:
                reject("analysis", str(exc), error=exc.as_dict())
            decisions_payload = canonical(analysis["input"]["decisions"])
            observations_payload = canonical(analysis["input"]["observations"])
            binding = {
                "release": {
                    "id": release,
                    "digest": release_record["digest"],
                    "state_at_binding": release_record["state"],
                    "approved_by": approval["reviewer"],
                    "approval_reason": approval["reason"],
                    "selected": manifest["selected"],
                    "version_count": len(manifest["versions"]),
                },
                "inputs": [
                    {
                        "dataset": record["dataset"],
                        "version": record["version"],
                        "version_digest": record["digest"],
                        "artifact": record["artifact"],
                        "artifact_bytes": self.artifacts.verify(record["artifact"])["bytes"],
                        "rows": record["row_count"],
                        "schema": {
                            "id": schema.id,
                            "version": schema.version,
                            "fingerprint": schema.fingerprint,
                            "definition": schema.as_dict(),
                        },
                        "state_at_binding": record["state"],
                    }
                ],
                "config": _public_config(parsed),
                "seed": parsed["seed"],
                "analysis_version": ANALYSIS_VERSION,
                "row_counts": {
                    "input": analysis["input"]["rows"],
                    "included": analysis["input"]["rows_included"],
                    "excluded": analysis["input"]["rows_excluded"],
                },
            }
            analysis_payload = canonical(analysis)
            record_digest = digest(
                {
                    "binding": binding,
                    "analysis": analysis,
                    "decisions": analysis["input"]["decisions"],
                    "observations": analysis["input"]["observations"],
                }
            )
            con.execute(
                "INSERT INTO life_experiments VALUES(?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    canonical(request),
                    canonical(binding),
                    analysis_payload,
                    decisions_payload,
                    observations_payload,
                    record_digest,
                ),
            )
            self.db.audit(
                "life_experiment_saved",
                experiment_id,
                {
                    "dataset": dataset,
                    "version": version,
                    "release": release,
                    "release_digest": release_record["digest"],
                    "digest": record_digest,
                    "status": analysis["status"],
                },
            )
        return self._record(experiment_id)

    # ------------------------------------------------------------- retrieval
    def get(self, experiment_id):
        name(experiment_id)
        return self._record(experiment_id)

    def list(self):
        return [
            {
                "id": row["id"],
                "dataset": parse_json(row["request"])["dataset"],
                "version": parse_json(row["request"])["version"],
                "release": parse_json(row["request"])["release"],
                "digest": row["digest"],
            }
            for row in self.db.rows("SELECT id,request,digest FROM life_experiments ORDER BY id")
        ]

    def _record(self, experiment_id):
        row = self.db.one("SELECT * FROM life_experiments WHERE id=?", (experiment_id,))
        if row is None:
            reject("missing", "life experiment not found")
        binding = parse_json(row["binding"])
        analysis = parse_json(row["analysis"])
        decisions = parse_json(row["decisions"])
        observations = parse_json(row["observations"])
        if (
            canonical(decisions) != row["decisions"]
            or canonical(observations) != row["observations"]
        ):
            reject("integrity", "life experiment row decisions were altered")
        stored = digest(
            {
                "binding": binding,
                "analysis": analysis,
                "decisions": decisions,
                "observations": observations,
            }
        )
        if stored != row["digest"]:
            reject("integrity", "life experiment record digest mismatch")
        suitability = self._suitability(binding)
        return {
            "id": experiment_id,
            "request": parse_json(row["request"]),
            "binding": binding,
            "analysis": analysis,
            "digest": row["digest"],
            **suitability,
        }

    def _suitability(self, binding):
        reasons = []
        release = binding["release"]
        current_release = self.releases.get(release["id"])
        if current_release["state"] == "revoked":
            reasons.append(
                {
                    "code": "release_revoked",
                    "message": "the originating release was revoked; this result is no longer suitable for delivery",
                }
            )
        elif current_release["digest"] != release["digest"]:
            reasons.append(
                {
                    "code": "release_changed",
                    "message": "the stored release manifest no longer matches the frozen digest",
                }
            )
        elif current_release["state"] != "approved":
            reasons.append(
                {
                    "code": "release_not_approved",
                    "message": "the originating release is no longer approved",
                    "state": current_release["state"],
                }
            )
        eligibility = self.releases.eligibility(release["id"])
        if not eligibility["eligible"]:
            reasons.extend(
                {
                    "code": "input_not_eligible",
                    "message": "a bound input version or artifact is no longer eligible",
                    "detail": problem,
                }
                for problem in eligibility["problems"]
            )
        for item in binding["inputs"]:
            current = self.registry.version(item["dataset"], item["version"])
            if current["state"] != "active":
                reasons.append(
                    {
                        "code": "input_version_withdrawn",
                        "message": "the bound dataset version was withdrawn",
                        "dataset": item["dataset"],
                        "version": item["version"],
                    }
                )
            if (
                current["digest"] != item["version_digest"]
                or current["artifact"] != item["artifact"]
            ):
                reasons.append(
                    {
                        "code": "input_version_changed",
                        "message": "the bound dataset version content no longer matches",
                        "dataset": item["dataset"],
                        "version": item["version"],
                    }
                )
            try:
                self.artifacts.verify(item["artifact"])
            except Exception:
                reasons.append(
                    {
                        "code": "input_artifact_unavailable",
                        "message": "the bound input artifact cannot be verified",
                        "artifact": item["artifact"],
                    }
                )
        return {
            "suitable_for_delivery": not reasons,
            "suitability_reasons": reasons,
            "recomputed_from_later_version": False,
            "bound_version": binding["inputs"][0]["version"],
        }

    # --------------------------------------------------------------- export
    def export(self, export_id, experiment_id, target):
        name(export_id)
        target = Path(target).absolute()
        if not target.parent.is_dir() or target.is_symlink() or target.parent.is_symlink():
            reject("path", "life export requires a safe existing parent")
        record = self._record(experiment_id)
        if not record["suitable_for_delivery"]:
            reject(
                "state",
                "life experiment is not suitable for delivery",
                reasons=record["suitability_reasons"],
            )
        stage = target.parent / (".vault-life-" + export_id)
        if target.exists():
            reject("conflict", "life export target already exists")
        if stage.exists() or stage.is_symlink():
            reject("conflict", "life export staging path exists")
        stage.mkdir()
        try:
            inputs = []
            for item in record["binding"]["inputs"]:
                content = self.artifacts.read(item["artifact"])
                if hashlib.sha256(content).hexdigest() != item["artifact"]:
                    raise Failure("integrity", "input artifact hash mismatch")
                filename = item["artifact"] + ".jsonl"
                (stage / filename).write_bytes(content)
                inputs.append({**item, "bundle_member": filename})
            decisions_bytes = (
                "\n".join(canonical(item) for item in record["analysis"]["input"]["decisions"])
                + "\n"
            ).encode()
            observations_bytes = (
                "\n".join(canonical(item) for item in record["analysis"]["input"]["observations"])
                + "\n"
            ).encode()
            analysis_bytes = (canonical(record["analysis"]) + "\n").encode()
            (stage / "decisions.jsonl").write_bytes(decisions_bytes)
            (stage / "observations.jsonl").write_bytes(observations_bytes)
            (stage / "analysis.json").write_bytes(analysis_bytes)
            manifest = {
                "schema": 1,
                "kind": "life-experiment-export",
                "experiment": experiment_id,
                "analysis_version": ANALYSIS_VERSION,
                "record_digest": record["digest"],
                "analysis_digest": digest(record["analysis"]),
                "decisions_digest": hashlib.sha256(decisions_bytes).hexdigest(),
                "observations_digest": hashlib.sha256(observations_bytes).hexdigest(),
                "release": {
                    "id": record["binding"]["release"]["id"],
                    "digest": record["binding"]["release"]["digest"],
                    "approved_by": record["binding"]["release"]["approved_by"],
                },
                "inputs": inputs,
                "config": record["binding"]["config"],
                "seed": record["binding"]["seed"],
                "row_counts": record["binding"]["row_counts"],
                "members": sorted(
                    [item["bundle_member"] for item in inputs]
                    + ["analysis.json", "decisions.jsonl", "observations.jsonl"]
                ),
            }
            envelope = {"digest": digest(manifest), "manifest": manifest}
            (stage / "manifest.json").write_text(canonical(envelope) + "\n")
            for path in stage.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            verified = verify_life_export(stage)
            os.rename(stage, target)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            if stage.exists():
                for path in stage.iterdir():
                    path.unlink(missing_ok=True)
                stage.rmdir()
            raise
        with self.db.transaction() as con:
            self.db.audit(
                "life_experiment_exported",
                export_id,
                {"experiment": experiment_id, "digest": verified["digest"]},
            )
        return {"export": export_id, "state": "complete", **verified}


def verify_life_export(path, recompute=False):
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        reject("path", "life export must be a regular directory")
    manifest_path = root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > 16000000
    ):
        reject("integrity", "life export manifest absent or invalid")
    envelope = parse_json(manifest_path.read_bytes())
    if not isinstance(envelope, dict) or set(envelope) != {"digest", "manifest"}:
        reject("input", "invalid life export envelope")
    manifest = envelope["manifest"]
    if digest(manifest) != envelope["digest"]:
        reject("integrity", "life export manifest digest mismatch")
    if manifest.get("schema") != 1 or manifest.get("kind") != "life-experiment-export":
        reject("integrity", "invalid life export manifest")
    members = set(manifest["members"])
    actual = {item.name for item in root.iterdir()}
    if actual != members | {"manifest.json"}:
        reject(
            "integrity",
            "life export member set differs",
            missing=sorted(members - actual),
            extra=sorted(actual - members - {"manifest.json"}),
        )
    input_rows = 0
    for item in manifest["inputs"]:
        member = root / item["bundle_member"]
        if member.is_symlink() or not member.is_file():
            reject("path", "life export input must be a regular file")
        content = member.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["artifact"]:
            reject("integrity", "life export input hash mismatch")
        rows = [parse_json(line) for line in content.decode("utf-8").splitlines() if line]
        if len(rows) != item["rows"]:
            reject("integrity", "life export input row count mismatch")
        input_rows += len(rows)
    analysis_path = root / "analysis.json"
    if analysis_path.is_symlink() or not analysis_path.is_file():
        reject("path", "life export analysis must be a regular file")
    analysis = parse_json(analysis_path.read_bytes())
    if digest(analysis) != manifest["analysis_digest"]:
        reject("integrity", "life export analysis digest mismatch")
    decisions_path = root / "decisions.jsonl"
    observations_path = root / "observations.jsonl"
    decisions_bytes = decisions_path.read_bytes()
    observations_bytes = observations_path.read_bytes()
    if hashlib.sha256(decisions_bytes).hexdigest() != manifest["decisions_digest"]:
        reject("integrity", "life export decisions digest mismatch")
    if hashlib.sha256(observations_bytes).hexdigest() != manifest["observations_digest"]:
        reject("integrity", "life export observations digest mismatch")
    decisions = [
        parse_json(line) for line in decisions_bytes.decode("utf-8").splitlines() if line
    ]
    observations = [
        parse_json(line)
        for line in observations_bytes.decode("utf-8").splitlines() if line
    ]
    rows_seen = [item["row"] for item in decisions]
    if sorted(rows_seen) != list(range(1, input_rows + 1)):
        reject(
            "integrity",
            "life export row decisions do not cover each input row exactly once",
        )
    included = sum(item["decision"] == "included" for item in decisions)
    if included != manifest["row_counts"]["included"]:
        reject("integrity", "life export included row count differs")
    if len(observations) != analysis["input"]["included_specimens"]:
        reject("integrity", "life export observation count differs")
    if recompute:
        contract = Schema(manifest["inputs"][0]["schema"]["definition"])
        rebuilt = run_analysis(
            [row for item in manifest["inputs"] for row in _rows_of(root, item)],
            contract,
            manifest["config"],
        )
        if digest(rebuilt) != manifest["analysis_digest"]:
            reject(
                "integrity",
                "life export analysis does not reproduce from bundled inputs and config",
            )
    return {
        "experiment": manifest["experiment"],
        "digest": envelope["digest"],
        "record_digest": manifest.get("record_digest"),
        "inputs": len(manifest["inputs"]),
        "input_rows": input_rows,
        "included_rows": included,
        "observations": len(observations),
        "release": manifest["release"]["id"],
        "release_digest": manifest["release"]["digest"],
        "recomputed": bool(recompute),
    }


def _rows_of(root, item):
    return [
        parse_json(line)
        for line in (root / item["bundle_member"]).read_text().splitlines()
        if line
    ]
