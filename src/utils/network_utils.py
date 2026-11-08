"""Transporte compartilhado e mensagens versionadas V2V/V2I.

O protocolo V2I usa dicionários com versão e tipo. UPLOAD, RESULT e APPLIED
identificam execução, RSU, encontro e cliente. UPLOAD e RESULT carregam pesos
em CPU; ERROR informa falhas e PING/PONG verificam a prontidão do serviço.
UPLOAD exige train_samples nos dois modos; V2I v2 rejeita envelopes v1 sem contagens.
"""

import logging
import math
import pickle
import socket
import struct
import time
from contextlib import suppress

logger = logging.getLogger(__name__)

V2I_PROTOCOL_VERSION = 2
V2V_PROTOCOL_VERSION = 1
V2I_MESSAGE_TYPES = frozenset({"UPLOAD", "RESULT", "APPLIED", "ERROR", "PING", "PONG"})


class V2IProtocolError(ValueError):
    """Mensagem V2I inválida ou associada a outro encontro."""


def validate_train_samples(value: object) -> int:
    """Exige uma contagem explícita de amostras, sem aceitar bool, float ou zero."""
    if type(value) is not int or value <= 0:
        raise ValueError("train_samples deve ser um inteiro positivo.")
    return value


def make_v2v_message(message_type: str, **fields) -> dict:
    """Envelope V2V versionado; UPLOAD informa amostras, RESULT carrega a média."""
    if {"version", "type"}.intersection(fields):
        raise ValueError("type e version são definidos pelo construtor V2V.")
    return validate_v2v_message(
        {"version": V2V_PROTOCOL_VERSION, "type": message_type, **fields}
    )


def validate_v2v_message(message: object, **expected) -> dict:
    """Rejeita uploads antigos e mensagens de outro encontro antes de usar seus pesos."""
    if not isinstance(message, dict):
        raise ValueError("Mensagem V2V deve ser um envelope versionado.")
    if type(message.get("version")) is not int or message["version"] != V2V_PROTOCOL_VERSION:
        raise ValueError("Versão V2V não suportada.")
    kind = message.get("type")
    if kind not in ("UPLOAD", "RESULT"):
        raise ValueError("Tipo V2V deve ser UPLOAD ou RESULT.")
    if not isinstance(message.get("run_id"), str) or not message["run_id"].strip():
        raise ValueError("run_id V2V deve ser uma string não vazia.")
    for field, minimum in (("client_id", 0), ("encounter_id", 1)):
        if type(message.get(field)) is not int or message[field] < minimum:
            raise ValueError(f"{field} V2V inválido.")
    allowed = {"version", "type", "run_id", "client_id", "encounter_id", "weights"}
    _validate_model_weights(message.get("weights"))
    if kind == "UPLOAD":
        allowed.add("train_samples")
        validate_train_samples(message.get("train_samples"))
    if set(message) - allowed:
        raise ValueError("Campos V2V não suportados.")
    for field, value in expected.items():
        if message.get(field) != value:
            raise ValueError(f"{field} V2V não corresponde ao encontro esperado.")
    return message


def _check_deadline(deadline: float | None, cancelled) -> float:
    """Verifica prazo/cancelamento e limita cada espera no socket a 200 ms."""
    if cancelled is not None and cancelled():
        raise ConnectionAbortedError("Operação de rede cancelada.")
    if deadline is None:
        return 0.2
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Prazo da operação de rede esgotado.")
    return min(remaining, 0.2)


def send_data(conn: socket.socket, data, *, deadline: float | None = None, cancelled=None) -> int:
    """Serializa e envia dados pelo socket com cabeçalho de tamanho.

    Retorna os bytes da mensagem completa (4 de cabeçalho + payload pickle).
    deadline é um prazo absoluto de time.monotonic(); cancelled é uma função
    sem argumentos que sinaliza cancelamento. Sem ambos, mantém o envio antigo.
    Envios parciais em caso de erro e retransmissões TCP não são contabilizados.
    """
    controlled = deadline is not None or cancelled is not None
    if controlled:
        _check_deadline(deadline, cancelled)
    data_bytes = pickle.dumps(data)
    header = struct.pack("!I", len(data_bytes))
    total_bytes = 4 + len(data_bytes)
    if not controlled:
        conn.sendall(header)
        conn.sendall(data_bytes)
        return total_bytes

    previous_timeout = conn.gettimeout()
    try:
        frame = memoryview(header + data_bytes)
        while frame:
            conn.settimeout(_check_deadline(deadline, cancelled))
            try:
                sent = conn.send(frame)
            except TimeoutError:
                continue
            if sent == 0:
                raise ConnectionError("Conexão encerrada durante o envio.")
            frame = frame[sent:]
        _check_deadline(deadline, cancelled)
        return total_bytes
    finally:
        with suppress(OSError):
            conn.settimeout(previous_timeout)


def recvall(
    conn: socket.socket, n: int, *, deadline: float | None = None, cancelled=None
) -> bytes | None:
    """Recebe exatamente n bytes, inclusive quando chegam em fragmentos.

    Sem controle adicional, mantém None para timeout ou desconexão. Com controle,
    prazo esgotado gera TimeoutError, cancelamento gera ConnectionAbortedError e
    desconexão gera ConnectionError. O timeout anterior do socket é restaurado.
    """
    controlled = deadline is not None or cancelled is not None
    previous_timeout = conn.gettimeout() if controlled else None
    data = bytearray()
    try:
        while len(data) < n:
            if controlled:
                conn.settimeout(_check_deadline(deadline, cancelled))
            try:
                packet = conn.recv(n - len(data))
            except TimeoutError:
                if controlled:
                    continue
                logger.warning("Socket timeout durante recebimento de dados.")
                return None
            if not packet:
                if controlled:
                    raise ConnectionError("Conexão encerrada antes da mensagem completa.")
                return None
            data.extend(packet)
        if controlled:
            _check_deadline(deadline, cancelled)
        return bytes(data)
    finally:
        if controlled:
            with suppress(OSError):
                conn.settimeout(previous_timeout)


def recv_data(
    conn: socket.socket,
    return_bytes: bool = False,
    *,
    deadline: float | None = None,
    cancelled=None,
):
    """Recebe dados via socket calculando a taxa de bits (bitrate).

    Se return_bytes=False (padrão): retorna (data, bit_rate).
    Se return_bytes=True: retorna (data, bit_rate, total_bytes) onde total_bytes = msglen + 4.
    O mesmo deadline e cancelled são usados para o cabeçalho e o conteúdo.
    Sem esses argumentos, preserva os retornos antigos em caso de falha.
    """
    controlled = deadline is not None or cancelled is not None
    clock = time.monotonic if controlled else time.time
    start_time = clock()

    raw_msglen = recvall(conn, 4, deadline=deadline, cancelled=cancelled)
    if not raw_msglen:
        return (None, 0, 0) if return_bytes else (None, 0)

    msglen = struct.unpack("!I", raw_msglen)[0]
    data_bytes = recvall(conn, msglen, deadline=deadline, cancelled=cancelled)
    end_time = clock()

    if not data_bytes and not controlled:
        return (None, 0, 0) if return_bytes else (None, 0)

    total_bytes = msglen + 4
    total_bits = total_bytes * 8
    total_time = end_time - start_time
    bit_rate = total_bits / total_time if total_time > 0 else 0

    data = pickle.loads(data_bytes)
    if controlled:
        _check_deadline(deadline, cancelled)
    if return_bytes:
        return data, bit_rate, total_bytes
    return data, bit_rate


def _validate_v2i_context(message: dict) -> None:
    for field in ("run_id", "rsu_id"):
        value = message.get(field)
        if not isinstance(value, str) or not value.strip():
            raise V2IProtocolError(f"{field} deve ser uma string não vazia.")

    encounter_id = message.get("encounter_id")
    if type(encounter_id) is not int or encounter_id < 1:
        raise V2IProtocolError("encounter_id deve ser um inteiro positivo.")

    if "client_id" in message:
        client_id = message["client_id"]
        if type(client_id) is not int or client_id < 0:
            raise V2IProtocolError("client_id deve ser um inteiro não negativo.")


def _validate_model_weights(weights: object) -> None:
    # Manter o transporte genérico e o healthcheck independentes do PyTorch.
    import torch

    if not isinstance(weights, dict) or not weights:
        raise V2IProtocolError("weights deve ser um state_dict não vazio.")
    for name, tensor in weights.items():
        if not isinstance(name, str) or not name:
            raise V2IProtocolError("As chaves de weights devem ser strings não vazias.")
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise V2IProtocolError(f"O peso {name!r} deve ser um tensor em CPU.")
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.is_complex():
            raise V2IProtocolError(f"O peso {name!r} deve ser um tensor real e denso.")
        if tensor.numel() == 0 or not torch.isfinite(tensor).all().item():
            raise V2IProtocolError(f"O peso {name!r} deve conter valores finitos e não vazios.")


def validate_v2i_message(message: object) -> dict:
    """Valida a estrutura sem alterar a mensagem ou os pesos recebidos.

    ERROR pode omitir o contexto para responder a uma mensagem ilegível. Se
    qualquer identificador for informado, o contexto completo é obrigatório;
    client_id é opcional apenas em ERROR. PING/PONG usam somente tipo e versão.
    Participação no encontro e compatibilidade entre modelos cabem ao servidor.
    """
    if not isinstance(message, dict):
        raise V2IProtocolError("A mensagem V2I deve ser um dicionário.")
    version = message.get("version")
    if type(version) is not int or version != V2I_PROTOCOL_VERSION:
        raise V2IProtocolError(f"Versão V2I não suportada: {version!r}.")
    message_type = message.get("type")
    if not isinstance(message_type, str) or message_type not in V2I_MESSAGE_TYPES:
        raise V2IProtocolError(f"Tipo V2I não suportado: {message_type!r}.")

    allowed = {"version", "type"}
    context = {"run_id", "rsu_id", "encounter_id", "client_id"}
    if message_type not in {"PING", "PONG"}:
        allowed |= context
        if message_type != "ERROR" or context.intersection(message):
            _validate_v2i_context(message)
        if message_type != "ERROR" and "client_id" not in message:
            raise V2IProtocolError("client_id é obrigatório nesta mensagem.")

    if message_type in {"UPLOAD", "RESULT"}:
        allowed.add("weights")
        _validate_model_weights(message.get("weights"))
    elif message_type == "ERROR":
        allowed.add("error")
        error = message.get("error")
        if not isinstance(error, str) or not error.strip():
            raise V2IProtocolError("ERROR deve informar uma descrição não vazia em error.")

    if message_type == "UPLOAD":
        allowed.add("train_samples")
        try:
            validate_train_samples(message.get("train_samples"))
        except ValueError as exc:
            raise V2IProtocolError(str(exc)) from exc

    if message_type == "RESULT":
        allowed.add("aggregation_time")
        duration = message.get("aggregation_time", 0.0)
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise V2IProtocolError("aggregation_time deve ser finito e não negativo.")

    unexpected = set(message) - allowed
    if unexpected:
        raise V2IProtocolError(f"Campos V2I não suportados: {list(unexpected)!r}.")
    return message


def make_v2i_message(message_type: str, **fields) -> dict:
    """Monta uma mensagem V2I validada com a versão atual.

    Exemplo: make_v2i_message("UPLOAD", run_id="run-1", rsu_id="rsu_0",
                             encounter_id=1, client_id=0, weights=state_dict, train_samples=100)
    """
    if {"version", "type"}.intersection(fields):
        raise V2IProtocolError("type e version são definidos pelo construtor V2I.")
    return validate_v2i_message({"version": V2I_PROTOCOL_VERSION, "type": message_type, **fields})


def send_v2i_message(
    conn: socket.socket, message: dict, *, deadline: float | None = None, cancelled=None
) -> int:
    """Envia uma mensagem validada e retorna bytes de payload mais o cabeçalho.

    O contador é retornado somente após o envio completo. Retransmissões TCP e
    bytes parcialmente enviados em caso de erro não são medidos por este retorno.
    deadline é um prazo absoluto monotônico, comum a toda a troca. cancelled
    permite interromper a espera mesmo enquanto o socket recebe ou envia dados.
    Sem prazo ou cancelamento, preserva o timeout do socket.
    """
    return send_data(
        conn,
        validate_v2i_message(message),
        deadline=deadline,
        cancelled=cancelled,
    )


def recv_v2i_message(
    conn: socket.socket,
    *,
    run_id: str | None = None,
    rsu_id: str | None = None,
    encounter_id: int | None = None,
    client_id: int | None = None,
    deadline: float | None = None,
    cancelled=None,
) -> tuple[dict, float, int]:
    """Recebe (mensagem, bitrate, bytes) e confere os identificadores esperados.

    Informe os identificadores ao aguardar RESULT/APPLIED para rejeitar respostas
    de outros encontros. Lança ConnectionError se a conexão encerrar antes da
    mensagem completa, TimeoutError se deadline expirar e V2IProtocolError para
    mensagens inválidas. Sem prazo/cancelamento, o timeout do socket vira ConnectionError.
    deadline e cancelled têm o mesmo significado de send_v2i_message.
    """
    try:
        message, bit_rate, total_bytes = recv_data(
            conn,
            return_bytes=True,
            deadline=deadline,
            cancelled=cancelled,
        )
    except (pickle.UnpicklingError, EOFError) as exc:
        raise V2IProtocolError("Payload V2I inválido.") from exc
    if message is None:
        raise ConnectionError("Conexão encerrada ou timeout antes da mensagem V2I completa.")
    validate_v2i_message(message)
    expected = {
        "run_id": run_id,
        "rsu_id": rsu_id,
        "encounter_id": encounter_id,
        "client_id": client_id,
    }
    for field, value in expected.items():
        if value is not None and message.get(field) != value:
            raise V2IProtocolError(
                f"{field} inesperado: esperado {value!r}, recebido {message.get(field)!r}."
            )
    return message, bit_rate, total_bytes
