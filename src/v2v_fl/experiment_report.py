"""Compara avaliações locais, cobertura dos clientes e erro OTA de execuções explícitas."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from v2v_fl.validate_mobility import contact_summary

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def weighted(records, field):
    valid = [r for r in records if isinstance(r.get(field), (int, float))
             and math.isfinite(r[field]) and r.get("test_samples", 0) > 0]
    samples = sum(r["test_samples"] for r in valid)
    return sum(r[field] * r["test_samples"] for r in valid) / samples if samples else None


def summarize_run(directory: Path) -> dict:
    metadata = json.loads((directory / "run.json").read_text())
    mode = metadata["mode"]
    total = metadata["parameters"]["total_clients"]
    path = directory / f"{mode}_events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = [e for e in events if e.get("type") == "encounter_formed"
              and e.get("status") == "completed"]
    info = metadata.get("dataset") or {}
    dataset_dir = PROJECT_ROOT / info.get(
        "directory", f"src/dataset/{metadata['parameters']['dataset']}"
    )
    baselines = {}
    for client in range(total):
        with np.load(dataset_dir / "test" / f"{client}.npz", allow_pickle=True) as archive:
            labels = archive["data"].item()["y"]
        classes, counts = np.unique(labels, return_counts=True)
        baselines[str(client)] = {
            "majority_class": int(classes[counts.argmax()]),
            "accuracy": float(100 * counts.max() / counts.sum()),
            "test_samples": len(labels),
        }
    records = []
    timeline = []
    for index, event in enumerate(events):
        current = []
        for record in event.get("client_metrics", []):
            if record.get("status") != "completed":
                continue
            record = dict(record)
            record["majority_class_baseline"] = baselines[str(record["client_id"])]["accuracy"]
            current.append(record)
        records.extend(current)
        timeline.append({
            "encounter_id": event["encounter_id"], "sim_time": event["sim_time"],
            "clients": event["clients"],
            "accuracy_before_weighted": weighted(current, "accuracy_before"),
            "accuracy_after_weighted": weighted(current, "accuracy_after"),
            "majority_class_baseline_weighted": weighted(current, "majority_class_baseline"),
            "ota_relative_l2_error": (event.get("ota_diagnostics") or {}).get("relative_l2_error"),
            "cumulative_components": contact_summary(events[:index + 1], total)["components"],
        })
    errors = [e["ota_relative_l2_error"] for e in timeline
              if isinstance(e["ota_relative_l2_error"], (int, float))]
    return {
        "run_id": metadata["run_id"], "mode": mode,
        "alpha": info.get("config", {}).get("alpha"), "dataset_directory": str(dataset_dir),
        "expected_encounters": metadata["parameters"]["encounters"],
        **contact_summary(events, total),
        "accuracy_before_weighted_participations": weighted(records, "accuracy_before"),
        "accuracy_after_weighted_participations": weighted(records, "accuracy_after"),
        "majority_class_baseline_weighted_participations": weighted(records, "majority_class_baseline"),
        "client_baselines": baselines,
        "ota_relative_l2_error_median": statistics.median(errors) if errors else None,
        "ota_relative_l2_error_max": max(errors) if errors else None,
        "timeline": timeline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    runs = [summarize_run(p.resolve()) for p in args.run_dir]
    scope = ("Acurácias locais das participações concluídas, ponderadas por amostras de teste; "
             "clientes podem ser contados em vários encontros. Os perfis possuem partições de "
             "teste diferentes. Não representa avaliação em um teste global comum.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"accuracy_scope": scope, "runs": runs}, indent=2) + "\n"
    )
    lines = ["# Comparação do recorte da Marambaia", "", scope, "",
             "| Execução | Modo | Alpha | Encontros | Antes (%) | Depois (%) | Majoritária (%) | Componentes |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    def fmt(value):
        return f"{value:.2f}" if value is not None else "—"
    for run in runs:
        lines.append(
            f"| {run['run_id']} | {run['mode'].upper()} | {run['alpha']} | "
            f"{run['encounter_count']}/{run['expected_encounters']} | "
            f"{fmt(run['accuracy_before_weighted_participations'])} | "
            f"{fmt(run['accuracy_after_weighted_participations'])} | "
            f"{fmt(run['majority_class_baseline_weighted_participations'])} | {run['components']} |"
        )
        print(f"{run['mode'].upper()} {run['run_id']}: {run['encounter_count']} encontros, "
              f"participação={run['participation']}")
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    if args.require_complete and any(r["encounter_count"] != r["expected_encounters"] for r in runs):
        parser.exit(1, "Execução com quantidade incompleta de encontros.\n")


if __name__ == "__main__":
    main()
