import json
import hashlib
import math
import os
import tempfile
import uuid
from pathlib import Path

import torch


def write_json_atomic(data: dict, path: Path) -> None:
    """Publica sinais e métricas completos, legíveis no host e nos containers."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, allow_nan=False)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dataset_manifest(project_root: Path, args) -> dict | None:
    """Identifica as partições realmente selecionadas, antes de criar a execução."""
    selected = getattr(args, "dataset_dir", None)
    dataset = getattr(args, "dataset", None)
    if selected is None and dataset is None:
        return None
    root = (project_root / "src/dataset").resolve()
    directory = Path(selected) if selected is not None else root / dataset
    if not directory.is_absolute():
        directory = project_root / directory
    directory = directory.resolve()
    if selected is None and not (directory / "config.json").exists():
        return None
    relative = directory.relative_to(root)
    config = json.loads((directory / "config.json").read_text())
    if getattr(args, "total_clients", 0) > config["num_clients"]:
        raise ValueError("Dataset possui menos partições que --total-clients.")
    files = [directory / "config.json"] + [
        directory / split / f"{client_id}.npz"
        for split in ("train", "test")
        for client_id in range(args.total_clients)
    ]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.relative_to(directory).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    if selected is not None:
        args.dataset_dir = directory
    return {"directory": f"src/dataset/{relative.as_posix()}", "config": config, "sha256": hashes}


def prepare_run(project_root: Path, mode: str, args) -> tuple[str, Path, Path]:
    """Isola resultados e sinais por execução; um ID existente nunca é sobrescrito."""
    run_id = getattr(args, "run_id", None) or uuid.uuid4().hex
    if not isinstance(run_id, str) or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id deve ser um nome de diretório simples.")
    dataset_info = dataset_manifest(project_root, args)
    directory = project_root / "src/results/runs" / mode / run_id
    directory.mkdir(parents=True, exist_ok=False)
    encounters_dir = directory / "encounters"
    encounters_dir.mkdir()
    (directory / "clients").mkdir()
    parameters = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    write_json_atomic(
        {
            "schema_version": 1,
            "run_id": run_id,
            "mode": mode,
            "parameters": {**parameters, "seed": getattr(args, "seed", 42)},
            "dataset": dataset_info,
            "metric_definitions": {
                "bytes": "Completed application frames including pickle envelope and "
                "4-byte header; "
                "excludes TCP/IP headers, retransmissions and incomplete frames. "
                "Model counters cover accepted uploads/results; health checks and rejected "
                "uploads are excluded.",
                "protocol_time": "Maximum client elapsed seconds until model application and "
                "required sends complete; excludes evaluation.",
                "aggregation_time": "Coordinator elapsed seconds executing the OTA pipeline.",
                "communication_wait_time": "Protocol time minus aggregation; includes waiting "
                "and model application, not a pure network latency measurement.",
                "evaluation_time": "Maximum client elapsed seconds evaluating the applied model.",
                "accuracy": "Percent accuracy on each client's test partition; mean and "
                "sample-weighted mean cover only completed evaluations.",
                "aggregation": "FedAvg weighted by len(train_loader.dataset), including the "
                "local V2V coordinator. OTA uses q_i=M*n_i/sum(n) over all M participants; "
                "final receiver noise std is sqrt(noise_variance)*norm(f)/M.",
                "ota_diagnostics": "Coordinator/RSU compares the delivered OTA model with the "
                "ideal weighted mean of the same inputs. relative_l2_error="
                "norm(OTA-ideal)/norm(ideal), concatenating all state_dict tensors in float64; "
                "null if norm(ideal)=0 (see error_l2_norm). Transmitter gains use H@conj(f); "
                "effective_model_weight=Re(b*true_gain)/sqrt(weight_variance)/M. "
                "noise_std describes the final delivered model, ota_noise_std the TX mean "
                "before the local V2V contribution. Ideal weights are diagnostic only.",
            },
        },
        directory / "run.json",
    )
    return run_id, directory, encounters_dir


def summarize_encounter_metrics(
    encounters_dir: Path, signal: dict, server_status: dict | None = None
) -> dict:
    """Consolida relatórios do encontro sem somar os dois extremos de um envio."""
    records = []
    encounter_id = signal["encounter_id"]
    for client_id in signal["clients"]:
        for name in (
            f".done_enc_{encounter_id}_client_{client_id}",
            f".failed_enc_{encounter_id}_client_{client_id}.json",
            f".applied_enc_{encounter_id}_client_{client_id}.json",
        ):
            try:
                record = json.loads((encounters_dir / name).read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            if any(record.get(key) != signal.get(key) for key in ("run_id", "encounter_id")):
                continue
            if record.get("client_id") != client_id or record.get("rsu_id") != signal.get(
                "rsu_id"
            ):
                continue
            records.append(record)
            break
    evaluated = [
        r
        for r in records
        if r.get("status") == "completed"
        and isinstance(r.get("accuracy_after"), (int, float))
        and math.isfinite(r["accuracy_after"])
    ]
    coordinator = next((r for r in records if r.get("is_aggregator")), None)
    source = server_status if signal.get("type") == "v2i" else coordinator
    byte_source = "rsu" if server_status is not None else "coordinator"
    counts_authoritative = bool(source)
    if not source:
        # Uma falha pode chegar antes do status terminal. Manter um limite inferior
        # com os relatórios disponíveis e marcar explicitamente a contagem parcial.
        byte_source = "clients_partial"
        source = {
            "upload_bytes": sum(r.get("upload_bytes", 0) for r in records),
            "download_bytes": sum(r.get("download_bytes", 0) for r in records),
            "received_upload_bytes": sum(r.get("upload_bytes", 0) for r in records),
            "sent_download_bytes": sum(r.get("download_bytes", 0) for r in records),
            "control_upload_bytes": sum(r.get("control_upload_bytes", 0) for r in records),
            "control_download_bytes": sum(r.get("control_download_bytes", 0) for r in records),
            "aggregation_time": max(
                (r.get("aggregation_time", 0.0) for r in records), default=0.0
            ),
        }
    aggregation_time = source.get("aggregation_time", 0.0)
    protocol_time = max((r.get("protocol_time", 0.0) for r in records), default=0.0)
    samples = sum(r.get("test_samples", 0) for r in evaluated)
    return {
        "participant_count": len(signal["clients"]),
        "reported_clients": sorted(r["client_id"] for r in records),
        "applied_clients": sorted(r["client_id"] for r in records if r.get("applied")),
        "evaluated_clients": sorted(r["client_id"] for r in evaluated),
        "evaluated_count": len(evaluated),
        "accuracy_mean": sum(r["accuracy_after"] for r in evaluated) / len(evaluated)
        if evaluated
        else None,
        "accuracy_weighted": sum(r["accuracy_after"] * r.get("test_samples", 0) for r in evaluated)
        / samples
        if samples
        else None,
        "test_samples": samples,
        "upload_bytes": source.get(
            "upload_bytes" if server_status is not None else "received_upload_bytes", 0
        ),
        "download_bytes": source.get(
            "download_bytes" if server_status is not None else "sent_download_bytes", 0
        ),
        "control_upload_bytes": source.get("control_upload_bytes", 0),
        "control_download_bytes": source.get("control_download_bytes", 0),
        "byte_count_source": byte_source,
        "byte_counts_complete": counts_authoritative
        and len(records) == len(signal["clients"])
        and all(r.get("status") == "completed" for r in records),
        "protocol_time": protocol_time,
        "aggregation_time": aggregation_time,
        "communication_wait_time": max(0.0, protocol_time - aggregation_time),
        "evaluation_time": max((r.get("evaluation_time", 0.0) for r in records), default=0.0),
        "ota_diagnostics": source.get("ota_diagnostics"),
        "client_metrics": records,
    }


def save_state_dict_atomic(state_dict: dict, path: Path) -> None:
    """Publish complete weights readable by both container and host users."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            torch.save(state_dict, tmp_file)
            tmp_file.flush()
            # mkstemp defaults to 0600; Docker and host may have different UIDs.
            os.fchmod(tmp_file.fileno(), 0o644)
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def dequantization(global_state):
    """Dequantizes a state dictionary."""
    dequantized_state_dict = {}
    for k, v in global_state.items():
        if isinstance(v, dict) and v.get("dtype") == "quantized_int8":
            # Recupera tensores quantizados
            scale = v["scale"]
            dequantized_state_dict[k] = v["weights"].float() * scale
        else:
            # Mantém tensores normais
            dequantized_state_dict[k] = v
    return dequantized_state_dict


def quantization(state_dict):
    """Quantizes a state dictionary to int8."""
    quantized_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            scale = torch.max(torch.abs(v)) / 127.0
            quantized_weights = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
            quantized_state_dict[k] = {
                "dtype": "quantized_int8",
                "scale": scale,
                "weights": quantized_weights,
            }
        else:
            quantized_state_dict[k] = v
    return quantized_state_dict
