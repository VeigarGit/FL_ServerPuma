import pickle
import struct

def send_data(conn, data):
    """Sends pickled data over a socket connection."""
    data_bytes = pickle.dumps(data)
    conn.sendall(struct.pack('!I', len(data_bytes)))
    conn.sendall(data_bytes)

def recv_data(conn):
    """Receives pickled data from a socket connection."""
    raw_msglen = recvall(conn, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!I', raw_msglen)[0]
    data_bytes = recvall(conn, msglen)
    return pickle.loads(data_bytes)

def recvall(conn, n):
    """Helper function to receive exactly 'n' bytes from a socket."""
    data = b'' 
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data