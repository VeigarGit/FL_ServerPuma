import socket
import pickle
import struct
import time
import logging

logger = logging.getLogger(__name__)

def send_data(conn: socket.socket, data) -> int:
    """Serializa e envia dados pelo socket com cabeçalho de tamanho.
    Retorna o total de bytes efetivamente enviados (4 bytes cabeçalho + payload pickle).
    """
    data_bytes = pickle.dumps(data)
    total_bytes = 4 + len(data_bytes)
    conn.sendall(struct.pack('!I', len(data_bytes)))
    conn.sendall(data_bytes)
    return total_bytes

def recvall(conn: socket.socket, n: int):
    """Garante o recebimento exato de 'n' bytes."""
    data = b'' 
    while len(data) < n:
        try:
            packet = conn.recv(n - len(data))
        except socket.timeout:
            logger.warning("Socket timeout durante recebimento de dados.")
            return None
        if not packet:
            return None
        data += packet
    return data

def recv_data(conn: socket.socket, return_bytes: bool = False):
    """Recebe dados via socket calculando a taxa de bits (bitrate).
    
    Se return_bytes=False (padrão): retorna (data, bit_rate).
    Se return_bytes=True: retorna (data, bit_rate, total_bytes) onde total_bytes = msglen + 4.
    """
    start_time = time.time()
    
    raw_msglen = recvall(conn, 4)
    if not raw_msglen:
        return (None, 0, 0) if return_bytes else (None, 0)
    
    msglen = struct.unpack('!I', raw_msglen)[0]
    data_bytes = recvall(conn, msglen)
    end_time = time.time()
    
    if not data_bytes:
        return (None, 0, 0) if return_bytes else (None, 0)
    
    total_bytes = msglen + 4
    total_bits = total_bytes * 8
    total_time = end_time - start_time
    bit_rate = total_bits / total_time if total_time > 0 else 0
    
    data = pickle.loads(data_bytes)
    if return_bytes:
        return data, bit_rate, total_bytes
    return data, bit_rate