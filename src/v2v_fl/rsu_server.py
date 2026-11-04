"""Um serviço TCP em CPU para todas as RSUs lógicas de uma execução V2I.

Uso a partir da raiz do projeto:
    PYTHONPATH=src uv run python -m v2v_fl.rsu_server --run-id run-1

O orquestrador publica encounter_<id>.json em --encounters-dir com os campos
run_id, type="v2i", rsu_id, encounter_id, clients, positions e exchange_timeout.
ota_epsilon e ota_bandwidth_hz são opcionais. A configuração Marambaia fornece
as posições e antenas das RSUs. Somente sinais e status usam arquivos.

Cada cliente mantém uma conexão: UPLOAD → RESULT → APPLIED. As mensagens usam
as funções V2I de utils.network_utils. PING → PONG verifica a prontidão. Há
somente um encontro ativo, com pelo menos dois veículos distintos; nenhum modelo
é persistido ou passado ao seguinte.
O ciclo é espera de uploads → agregação → espera de APPLIED → encerramento.
O cálculo OTA já iniciado termina em CPU, mas um resultado fora do prazo é
descartado. As conexões e os tensores são liberados antes do próximo encontro.
Cliente, orquestrador e Compose serão conectados ao serviço em etapas seguintes.
"""

import argparse
import json
import logging
import math
import socket
import socketserver
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from utils.model_utils import write_json_atomic
from utils.network_utils import (
    V2IProtocolError,
    make_v2i_message,
    recv_v2i_message,
    send_v2i_message,
)
from v2v_fl.over_the_air.ota_aggregation import (
    DEFAULT_OTA_BANDWIDTH_HZ,
    DEFAULT_OTA_EPSILON,
    ota_aggregate,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} deve ser numérico.")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} deve ser finito{' e positivo' if positive else ''}.")
    return float(value)


def load_rsus(path: Path) -> dict[str, dict]:
    """Carrega apenas o cadastro das RSUs, sem criar processos ou modelos."""
    with path.open() as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("rsus"), list) or not data["rsus"]:
        raise ValueError("A configuração deve conter uma lista rsus não vazia.")
    rsus = {}
    for entry in data["rsus"]:
        if not isinstance(entry, dict):
            raise ValueError("Cada RSU deve ser um dicionário.")
        rsu_id = entry.get("id")
        if not isinstance(rsu_id, str) or not rsu_id.strip() or rsu_id in rsus:
            raise ValueError("Cada RSU deve ter um id não vazio e único.")
        antennas = entry.get("n_antennas", 16)
        if type(antennas) is not int or antennas < 1:
            raise ValueError("n_antennas deve ser um inteiro positivo.")
        rsus[rsu_id] = {
            "id": rsu_id,
            "x": finite_number(entry.get("x"), "x"),
            "y": finite_number(entry.get("y"), "y"),
            "radius": finite_number(entry.get("radius"), "radius", positive=True),
            "n_antennas": antennas,
        }
    return rsus


@dataclass
class Encounter:
    run_id: str
    rsu_id: str
    encounter_id: int
    clients: frozenset[int]
    positions: dict[int, tuple[float, float]]
    timeout: float
    epsilon: float
    bandwidth_hz: float
    started: float = field(default_factory=time.monotonic)
    weights: dict[int, dict] = field(default_factory=dict)
    train_samples: dict[int, int] = field(default_factory=dict)
    ota_diagnostics: dict | None = None
    received: set[int] = field(default_factory=set)
    applied: set[int] = field(default_factory=set)
    result: dict | None = None
    error: str | None = None
    upload_bytes: int = 0
    download_bytes: int = 0
    control_upload_bytes: int = 0
    control_download_bytes: int = 0
    aggregation_time: float = 0.0
    phase: str = "waiting_uploads"
    connections: dict[int, socket.socket] = field(default_factory=dict, repr=False)
    finished: float | None = None

    @property
    def deadline(self) -> float:
        return self.started + self.timeout

    def context(self, client_id: int) -> dict:
        return {
            "run_id": self.run_id,
            "rsu_id": self.rsu_id,
            "encounter_id": self.encounter_id,
            "client_id": client_id,
        }


class RSUServer(socketserver.ThreadingTCPServer):
    """Um processo agrega; threads atendem os sockets do encontro ativo."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        run_id: str,
        rsu_config: Path,
        encounters_dir: Path,
        exchange_timeout: float = 300.0,
    ):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id deve ser uma string não vazia.")
        self.run_id = run_id
        self.rsus = load_rsus(rsu_config)
        self.encounters_dir = Path(encounters_dir)
        self.exchange_timeout = finite_number(exchange_timeout, "exchange_timeout", positive=True)
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.active: Encounter | None = None
        self.last_encounter_id = 0
        super().__init__(address, RSURequestHandler)

    def _load_encounter(self, data: dict) -> Encounter:
        rsu_id = data.get("rsu_id")
        if not isinstance(rsu_id, str) or rsu_id not in self.rsus:
            raise ValueError("RSU desconhecida no sinal de encontro.")
        encounter_id = data.get("encounter_id")
        if type(encounter_id) is not int or encounter_id < 1:
            raise ValueError("encounter_id deve ser um inteiro positivo.")
        clients = data.get("clients")
        if (
            not isinstance(clients, list)
            or len(clients) < 2
            or any(type(c) is not int or c < 0 for c in clients)
            or len(set(clients)) != len(clients)
        ):
            raise ValueError("clients deve conter pelo menos dois índices únicos e não negativos.")
        raw_positions = data.get("positions")
        if not isinstance(raw_positions, dict):
            raise ValueError("positions deve ser um dicionário.")
        positions = {}
        for client_id in clients:
            point = raw_positions.get(str(client_id), raw_positions.get(client_id))
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Posição ausente ou inválida para cliente {client_id}.")
            positions[client_id] = (finite_number(point[0], "x"), finite_number(point[1], "y"))
        epsilon = finite_number(data.get("ota_epsilon", DEFAULT_OTA_EPSILON), "ota_epsilon")
        if not 0 <= epsilon < 1:
            raise ValueError("ota_epsilon deve estar no intervalo [0, 1).")
        return Encounter(
            run_id=self.run_id,
            rsu_id=rsu_id,
            encounter_id=encounter_id,
            clients=frozenset(clients),
            positions=positions,
            timeout=min(
                self.exchange_timeout,
                finite_number(
                    data.get("exchange_timeout", self.exchange_timeout),
                    "exchange_timeout",
                    positive=True,
                ),
            ),
            epsilon=epsilon,
            bandwidth_hz=finite_number(
                data.get("ota_bandwidth_hz", DEFAULT_OTA_BANDWIDTH_HZ),
                "ota_bandwidth_hz",
                positive=True,
            ),
        )

    def _next_encounter(self) -> Encounter | None:
        candidates = []
        for path in self.encounters_dir.glob("encounter_*.json"):
            suffix = path.stem.removeprefix("encounter_")
            if suffix.isdecimal() and int(suffix) > self.last_encounter_id:
                candidates.append((int(suffix), path))
        for encounter_id, path in sorted(candidates):
            try:
                with path.open() as stream:
                    data = json.load(stream)
                if not isinstance(data, dict):
                    raise ValueError(f"Sinal inválido: {path}")
                if data.get("run_id") != self.run_id or data.get("type") != "v2i":
                    continue
                if (
                    type(data.get("encounter_id")) is not int
                    or data["encounter_id"] != encounter_id
                ):
                    raise ValueError(f"ID do sinal difere do nome do arquivo: {path}")
                return self._load_encounter(data)
            except (OSError, ValueError) as exc:
                # Um sinal defeituoso não pode encerrar o serviço de todas as RSUs.
                self.last_encounter_id = encounter_id
                logger.warning("Sinal do encontro %d rejeitado: %s", encounter_id, exc)
                self._write_status(
                    encounter_id,
                    {
                        "run_id": self.run_id,
                        "encounter_id": encounter_id,
                        "status": "failed",
                        "error": str(exc),
                        "received_clients": [],
                        "applied_clients": [],
                        "upload_bytes": 0,
                        "download_bytes": 0,
                        "exchange_time": 0.0,
                    },
                )
        return None

    def _check_encounter(self, encounter: Encounter) -> None:
        """Chamado sob a condição, inclusive após o cálculo OTA em CPU."""
        if encounter.error:
            raise RuntimeError(encounter.error)
        if self.active is not encounter:
            raise RuntimeError("O encontro não está mais ativo.")
        if self.stopping.is_set() or (self.encounters_dir / ".simulation_done").exists():
            raise RuntimeError("Simulação encerrada.")
        if time.monotonic() >= encounter.deadline:
            raise TimeoutError("Prazo do encontro esgotado.")

    def _wait_for(self, encounter: Encounter, predicate) -> None:
        with self.condition:
            while True:
                self._check_encounter(encounter)
                if predicate():
                    return
                remaining = encounter.deadline - time.monotonic()
                self.condition.wait(timeout=min(remaining, 0.1))

    def _publish_status(self, encounter: Encounter, status: str) -> None:
        data = {
            "run_id": self.run_id,
            "rsu_id": encounter.rsu_id,
            "encounter_id": encounter.encounter_id,
            "status": status,
            "clients": sorted(encounter.clients),
            "received_clients": sorted(encounter.received),
            "applied_clients": sorted(encounter.applied),
            "error": encounter.error,
            "upload_bytes": encounter.upload_bytes,
            "download_bytes": encounter.download_bytes,
            "control_upload_bytes": encounter.control_upload_bytes,
            "control_download_bytes": encounter.control_download_bytes,
            "aggregation_time": encounter.aggregation_time,
            "ota_diagnostics": encounter.ota_diagnostics,
            "exchange_time": round(encounter.finished - encounter.started, 6),
        }
        self._write_status(encounter.encounter_id, data)

    def _write_status(self, encounter_id: int, data: dict) -> None:
        destination = self.encounters_dir / f"rsu_status_enc_{encounter_id}.json"
        write_json_atomic(data, destination)

    def _process_encounter(self, encounter: Encounter) -> None:
        status = "completed"
        with self.condition:
            self.active = encounter
            self.condition.notify_all()
        logger.info(
            "Encontro %d na %s: clientes %s",
            encounter.encounter_id,
            encounter.rsu_id,
            sorted(encounter.clients),
        )
        result = None
        try:
            self._wait_for(encounter, lambda: encounter.received == encounter.clients)
            with self.condition:
                self._check_encounter(encounter)
                encounter.phase = "aggregating"
            aggregation_started = time.monotonic()
            diagnostics = {}
            try:
                result = ota_aggregate(
                    encounter.weights,
                    encounter.positions,
                    self.rsus[encounter.rsu_id],
                    train_samples=encounter.train_samples,
                    diagnostics=diagnostics,
                    epsilon=encounter.epsilon,
                    bandwidth_hz=encounter.bandwidth_hz,
                )
            finally:
                encounter.aggregation_time = time.monotonic() - aggregation_started
            with self.condition:
                # O cálculo em CPU pode terminar após o prazo ou após stop().
                self._check_encounter(encounter)
                encounter.ota_diagnostics = diagnostics
                encounter.weights.clear()
                encounter.result = result
                result = None
                encounter.phase = "waiting_applied"
                self.condition.notify_all()
            self._wait_for(encounter, lambda: encounter.applied == encounter.clients)
        except Exception as exc:
            expired = isinstance(exc, TimeoutError) or time.monotonic() >= encounter.deadline
            status = "timed_out" if expired else "failed"
            with self.condition:
                encounter.error = encounter.error or str(exc)
            logger.warning("Encontro %d: %s", encounter.encounter_id, exc)
        except KeyboardInterrupt:
            status = "failed"
            encounter.error = "Servidor interrompido."
            raise
        finally:
            result = None
            with self.condition:
                encounter.phase = status
                encounter.finished = time.monotonic()
                self.condition.notify_all()
                # Deixar os handlers em espera enviarem ERROR antes de interromper
                # sockets bloqueados num RESULT parcial ou na espera de APPLIED.
                self.condition.wait_for(lambda: not encounter.connections, timeout=0.1)
                for connection in list(encounter.connections.values()):
                    with suppress(OSError):
                        connection.shutdown(socket.SHUT_RDWR)
                # Nenhum handler deve reter pesos ou alterar status no próximo encontro.
                self.condition.wait_for(lambda: not encounter.connections)
                encounter.weights.clear()
                encounter.result = None
                self.last_encounter_id = encounter.encounter_id
                self.active = None
                self.condition.notify_all()
            self._publish_status(encounter, status)

    def accept_upload(
        self, message: dict, received_bytes: int, connection: socket.socket
    ) -> Encounter:
        if message["run_id"] != self.run_id or message["rsu_id"] not in self.rsus:
            raise V2IProtocolError("Execução ou RSU desconhecida.")
        with self.condition:
            # O cliente pode ler o JSON antes do próximo tick do coordenador.
            activation_deadline = time.monotonic() + min(5.0, self.exchange_timeout)
            while self.active is None and message["encounter_id"] > self.last_encounter_id:
                path = self.encounters_dir / f"encounter_{message['encounter_id']}.json"
                if not path.exists() or self.stopping.is_set():
                    break
                try:
                    with path.open() as stream:
                        signal = json.load(stream)
                except (OSError, ValueError):
                    break
                if not isinstance(signal, dict) or signal.get("run_id") != self.run_id:
                    break
                remaining = activation_deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(remaining, 0.1))
            encounter = self.active
            if encounter is None or any(
                message[key] != value
                for key, value in (
                    ("rsu_id", encounter.rsu_id),
                    ("encounter_id", encounter.encounter_id),
                )
            ):
                raise V2IProtocolError("O encontro solicitado não está ativo.")
            if (
                encounter.phase != "waiting_uploads"
                or encounter.error
                or self.stopping.is_set()
                or (self.encounters_dir / ".simulation_done").exists()
                or time.monotonic() >= encounter.deadline
            ):
                raise V2IProtocolError("O encontro já foi encerrado ou expirou.")
            client_id = message["client_id"]
            if client_id not in encounter.clients or client_id in encounter.received:
                raise V2IProtocolError("Cliente ausente do encontro ou upload duplicado.")
            state = message["weights"]
            if encounter.weights:
                reference = next(iter(encounter.weights.values()))
                if state.keys() != reference.keys() or any(
                    state[k].shape != reference[k].shape or state[k].dtype != reference[k].dtype
                    for k in reference
                ):
                    raise V2IProtocolError("Pesos incompatíveis com os outros participantes.")
            encounter.weights[client_id] = state
            encounter.train_samples[client_id] = message["train_samples"]
            encounter.connections[client_id] = connection
            encounter.received.add(client_id)
            encounter.upload_bytes += received_bytes
            self.condition.notify_all()
            return encounter

    def stop(self) -> None:
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()

    def run(self) -> None:
        """Serve os sockets em uma thread fixa; agrega sequencialmente na atual."""
        self.encounters_dir.mkdir(parents=True, exist_ok=True)
        listener = threading.Thread(target=self.serve_forever, daemon=True)
        listener.start()
        logger.info("RSUs %s disponíveis em %s:%d", list(self.rsus), *self.server_address)
        try:
            while not self.stopping.is_set():
                if (self.encounters_dir / ".simulation_done").exists():
                    break
                encounter = self._next_encounter()
                if encounter is None:
                    self.stopping.wait(0.1)
                else:
                    self._process_encounter(encounter)
        finally:
            self.stop()
            self.shutdown()
            listener.join(timeout=2)


class RSURequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        connection = self.request
        connection.settimeout(server.exchange_timeout)
        encounter = None
        message = result = ack = None
        context = {}
        try:
            message, _, received = recv_v2i_message(connection)
            if message["type"] == "PING":
                send_v2i_message(connection, make_v2i_message("PONG"))
                return
            if message["type"] != "UPLOAD":
                raise V2IProtocolError("Esperado UPLOAD ou PING.")
            context = {
                key: message[key] for key in ("run_id", "rsu_id", "encounter_id", "client_id")
            }
            encounter = server.accept_upload(message, received, connection)
            message = None
            server._wait_for(encounter, lambda: encounter.result is not None)
            with server.condition:
                server._check_encounter(encounter)
                result = encounter.result
            connection.settimeout(max(0.001, encounter.deadline - time.monotonic()))
            sent = send_v2i_message(
                connection,
                make_v2i_message(
                    "RESULT",
                    **context,
                    weights=result,
                    aggregation_time=encounter.aggregation_time,
                ),
            )
            result = None
            with server.condition:
                encounter.download_bytes += sent
            connection.settimeout(max(0.001, encounter.deadline - time.monotonic()))
            ack, _, control_bytes = recv_v2i_message(connection, **context)
            with server.condition:
                encounter.control_upload_bytes += control_bytes
            if ack["type"] != "APPLIED":
                raise V2IProtocolError("Esperado APPLIED após o resultado.")
            with server.condition:
                server._check_encounter(encounter)
                encounter.applied.add(ack["client_id"])
                server.condition.notify_all()
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            if encounter is not None:
                with server.condition:
                    if encounter.phase in {"waiting_uploads", "aggregating", "waiting_applied"}:
                        encounter.error = encounter.error or reason
                        server.condition.notify_all()
            logger.warning("Conexão %s: %s", self.client_address, exc)
            try:
                connection.settimeout(1)
                control_bytes = send_v2i_message(
                    connection, make_v2i_message("ERROR", **context, error=reason)
                )
                if encounter is not None:
                    with server.condition:
                        encounter.control_download_bytes += control_bytes
            except (OSError, V2IProtocolError):
                pass
        finally:
            # Soltar também referências locais antes de liberar o próximo encontro.
            message = result = ack = None
            connection.close()
            if encounter is not None:
                with server.condition:
                    encounter.connections.pop(context["client_id"], None)
                    server.condition.notify_all()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--rsu-config",
        type=Path,
        default=PROJECT_ROOT / "src/v2v_fl/maps/rsu_config_marambaia.json",
    )
    parser.add_argument(
        "--encounters-dir", type=Path, default=PROJECT_ROOT / "src/results/encounters"
    )
    parser.add_argument("--exchange-timeout", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=1, help="Threads de cálculo do PyTorch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("--port deve estar entre 1 e 65535; --threads deve ser positivo.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        with RSUServer(
            (args.host, args.port),
            run_id=args.run_id,
            rsu_config=args.rsu_config,
            encounters_dir=args.encounters_dir,
            exchange_timeout=args.exchange_timeout,
        ) as server:
            server.run()
    except KeyboardInterrupt:
        logger.info("Servidor encerrado.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
