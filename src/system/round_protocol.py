"""Mensagens e controle de prazo do FL centralizado com --timeout."""

import logging
import pickle
import socket
import time
from collections import deque
from contextlib import suppress
from queue import Empty, Full, Queue
from threading import Condition, Thread

from utils.network_utils import recv_payload, send_frame

logger = logging.getLogger(__name__)
PROTOCOL = "fl-round-v1"
CONTROL_TIMEOUT = 30.0


def round_message(kind, round_id, **fields):
    return {"protocol": PROTOCOL, "type": kind, "round": round_id, **fields}


def is_round_message(value, kind=None):
    return (
        isinstance(value, dict)
        and value.get("protocol") == PROTOCOL
        and type(value.get("round")) is int
        and (kind is None or value.get("type") == kind)
    )


def close_connection(conn):
    with suppress(OSError):
        conn.shutdown(socket.SHUT_RDWR)


class RoundInbox:
    """Registra frames completos antes da desserialização e em ordem de chegada."""

    def __init__(self):
        self.condition = Condition()
        self.frames = deque()
        self.disconnected = set()

    def receive(self, conn, client_id):
        try:
            while True:
                payload, _, size = recv_payload(conn, retry_timeout=True)
                if payload is None:
                    break
                with self.condition:
                    self.frames.append((time.monotonic(), client_id, payload, size))
                    self.condition.notify()
        except OSError:
            pass
        finally:
            with self.condition:
                self.disconnected.add(client_id)
                self.condition.notify()

    def get(self, deadline, expected):
        with self.condition:
            while not self.frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or expected <= self.disconnected:
                    return None
                self.condition.wait(remaining)
            if self.frames[0][0] > deadline:
                return None
            received, client_id, payload, size = self.frames.popleft()
        try:
            message = pickle.loads(payload)
        except (pickle.PickleError, EOFError, ValueError, RuntimeError):
            logger.warning("Cliente %s: frame inválido ignorado.", client_id)
            message = None
        return received, client_id, message, size


class RoundSender:
    """Mantém a ordem das mensagens e isola a escrita de cada cliente."""

    def __init__(self, conn, client_id):
        self.conn = conn
        self.client_id = client_id
        self.frames = Queue(maxsize=3)
        Thread(target=self.run, daemon=True).start()

    def send(self, frame):
        try:
            self.frames.put_nowait(frame)
        except Full:
            logger.warning("Cliente %s: fila de controle cheia.", self.client_id)
            close_connection(self.conn)

    def run(self):
        try:
            while True:
                frame = self.frames.get()
                if frame is None:
                    return
                send_frame(self.conn, frame)
        except OSError as exc:
            logger.warning("Cliente %s: falha no envio de controle: %s", self.client_id, exc)
            close_connection(self.conn)

    def close(self):
        close_connection(self.conn)
        self.send(None)


class RoundCancelled(Exception):
    """O treino local chegou a um ponto em que pode parar com segurança."""


class RoundControl:
    """Recebe cancelamentos enquanto a thread principal treina."""

    def __init__(self, conn, wait_timeout):
        self.conn = conn
        self.wait_timeout = wait_timeout
        self.messages = Queue()
        self.cancelled_round = 0
        self.disconnected = False
        self.receiver = Thread(target=self.receive, daemon=True)
        self.receiver.start()

    def close(self):
        close_connection(self.conn)
        self.receiver.join()

    def receive(self):
        try:
            while True:
                payload, _, _ = recv_payload(self.conn, retry_timeout=True)
                if payload is None:
                    break
                message = pickle.loads(payload)
                if is_round_message(message, "cancel"):
                    self.cancelled_round = message["round"]
                else:
                    self.messages.put(message)
        except (OSError, pickle.PickleError, EOFError):
            pass
        finally:
            self.disconnected = True
            self.messages.put(None)

    def expect(self, kind, round_id):
        try:
            message = self.messages.get(timeout=self.wait_timeout)
        except Empty as exc:
            raise TimeoutError(f"Servidor não enviou {kind} da rodada {round_id}.") from exc
        if not is_round_message(message, kind) or message["round"] != round_id:
            raise ConnectionError(f"Esperado {kind} da rodada {round_id}, conexão ou mensagem inválida.")
        return message

    def checkpoint(self, round_id, deadline):
        if self.disconnected:
            raise ConnectionError("Servidor desconectado durante o treino.")
        if self.cancelled_round >= round_id or time.monotonic() >= deadline:
            raise RoundCancelled


class CheckedBatches:
    """Executa um checkpoint entre batches sem alterar o DataLoader original."""

    def __init__(self, loader, checkpoint):
        self.loader = loader
        self.checkpoint = checkpoint

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        batches = iter(self.loader)
        while True:
            self.checkpoint()
            try:
                batch = next(batches)
            except StopIteration:
                return
            self.checkpoint()
            yield batch
