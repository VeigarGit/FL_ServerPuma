import socket
import pickle
import struct
import time
import logging

logger = logging.getLogger(__name__)

def send_data(conn: socket.socket, data):
    """Serializa e envia dados pelo socket com cabeçalho de tamanho."""
    data_bytes = pickle.dumps(data)
    conn.sendall(struct.pack('!I', len(data_bytes)))
    conn.sendall(data_bytes)

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

def recv_data(conn: socket.socket):
    """Recebe dados via socket calculando a taxa de bits (bitrate)."""
    start_time = time.time()
    
    raw_msglen = recvall(conn, 4)
    if not raw_msglen:
        return None, 0
    
    msglen = struct.unpack('!I', raw_msglen)[0]
    data_bytes = recvall(conn, msglen)
    end_time = time.time()
    
    if not data_bytes:
        return None, 0
    
    total_bits = (msglen + 4) * 8
    total_time = end_time - start_time
    bit_rate = total_bits / total_time if total_time > 0 else 0
    
    return pickle.loads(data_bytes), bit_rate