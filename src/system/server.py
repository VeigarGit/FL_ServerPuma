import socket
import threading
import argparse
import sys
import time
import signal
import numpy as np

# Import internal modules
from ..utils.network_utils import send_data, recv_data
from ..utils.logger import setup_logger
from ..utils.model_utils import dequantization
from .strategy import PUMAStrategy

# Initialize logger
logger = setup_logger("server.main")

class FederatedLearningServer:
    def __init__(self, args):
        self.args = args
        self.strategy = PUMAStrategy(args)
        
        # State Management
        self.stop_event = threading.Event()
        self.client_connections = []
        self.client_addresses = []
        self.client_ids = []
        self.lock = threading.Lock()
        
        # Round Data Holders
        self.current_round_updates = []
        self.current_round_metrics = [] # Stores {client_id: metrics_dict}
        
        # Synchronization Barriers
        self.start_round_barrier = threading.Barrier(args.clients_per_round + 1)
        self.end_round_barrier = threading.Barrier(args.clients_per_round + 1)

    def _profiling(self, metrics_list):
        """
        Analyzes client metrics to determine system heterogeneity.
        """
        client_times = {}
        client_volumes = {}
        
        logger.info(f"--- Profiling Round Data ({len(metrics_list)} clients) ---")
        
        for m in metrics_list:
            cid = m['client_id']
            train_time = m['training_time']
            comm_time = m['download_time']
            samples = m['samples']
            
            # Estimating bandwidth based on model size (approx 12MB) / download time
            estimated_payload_mb = 12.4 
            bw = estimated_payload_mb / comm_time if comm_time > 0 else 0
            
            logger.info(f"Profiled Client {cid}: bandwidth={bw:.2f} MB/s, train={train_time:.2f}s, samples={samples}")
            
            client_times[cid] = train_time
            client_volumes[cid] = samples

        return client_times, client_volumes

    def handle_client(self, conn, addr, idx):
        """
        Thread function to handle a single client's lifecycle.
        """
        client_id = None
        try:
            # 1. Handshake
            client_id = recv_data(conn)
            with self.lock:
                self.client_ids.append(client_id)
            logger.info(f"Client {client_id} registered from {addr}")
            
            # 2. Confirm Registration
            send_data(conn, "WAITING")
            
            # 3. Main Training Loop
            for round_num in range(self.args.rounds):
                # Wait for server signal to start round
                self.start_round_barrier.wait()
                
                if self.stop_event.is_set():
                    break
                
                # A. Get Model & Prune Rate from Strategy
                payload = self.strategy.get_model_package(client_id)
                
                # B. Send (Model, PruneRate)
                send_data(conn, payload)
                
                # C. Receive (Update, Metrics)
                response = recv_data(conn)
                if response is None:
                    logger.error(f"Client {client_id} disconnected unexpectedly.")
                    break
                    
                client_update_quantized, client_metrics = response
                
                # Convert the compressed dictionary back to float tensors
                # so the strategy can average them mathematically.
                client_update = dequantization(client_update_quantized)
                
                with self.lock:
                    self.current_round_updates.append(client_update)
                    self.current_round_metrics.append(client_metrics)
                    logger.info(f"Received update from Client {client_id} (Round {round_num})")
                
                # Wait for other clients to finish
                self.end_round_barrier.wait()
                
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            conn.close()

    def run_server(self):
        logger.info("--- FL Server starting... ---")
        logger.info(f"Strategy: PUMA+FedALA")
        logger.info(f"Host: {self.args.host}:{self.args.port}")
        
        # Setup Signal Handler for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received.")
            self.stop_event.set()
            try:
                self.start_round_barrier.reset()
                self.end_round_barrier.reset()
            except:
                pass
        
        signal.signal(signal.SIGINT, signal_handler)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.args.host, self.args.port))
            s.listen(self.args.clients_per_round)
            
            logger.info(f"Waiting for {self.args.clients_per_round} clients to connect...")
            
            # 1. Connection Phase
            threads = []
            while len(self.client_connections) < self.args.clients_per_round:
                if self.stop_event.is_set(): break
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                    self.client_connections.append(conn)
                    self.client_addresses.append(addr)
                    
                    # Start Client Thread
                    t = threading.Thread(target=self.handle_client, args=(conn, addr, len(self.client_connections)))
                    t.start()
                    threads.append(t)
                    
                    logger.info(f"Connected: {len(self.client_connections)}/{self.args.clients_per_round}")
                except socket.timeout:
                    continue
            
            if self.stop_event.is_set(): return

            # 2. Training Phase
            for round_num in range(self.args.rounds):
                logger.info(f"--- Starting Round {round_num} ---")
                
                # Reset round storage
                self.current_round_updates = []
                self.current_round_metrics = []
                
                # Release Clients to start training
                start_time = time.time()
                self.start_round_barrier.wait()
                
                # Wait for Clients to finish (Synchronization)
                self.end_round_barrier.wait()
                round_duration = time.time() - start_time
                
                logger.info(f"--- Round {round_num} Aggregation Phase ---")
                logger.info(f"Round duration: {round_duration:.2f}s")
                
                # 3. Aggregation & Strategy
                if self.current_round_updates:
                    # A. Profile Clients
                    client_times, client_volumes = self._profiling(self.current_round_metrics)
                    
                    # B. Aggregate
                    self.strategy.aggregate_updates(self.current_round_updates)
                    
                    # C. Evaluate
                    acc, loss = self.strategy.evaluate_global_model(round_num)
                    
                    # D. PUMA Pruning Trigger (Round 1 -> 2 transition)
                    if round_num == 1 and not self.strategy.prune_triggered:
                        logger.info("Triggering PUMA Adaptive Pruning...")
                        self.strategy.attempt_pruning(client_times, client_volumes)
                
                else:
                    logger.warning(f"Round {round_num}: No updates received.")

            logger.info("--- Training Complete. Sending shutdown signal... ---")
            
            # Notify clients to shutdown
            for conn in self.client_connections:
                try:
                    send_data(conn, "shutdown")
                except:
                    pass
            
            # Wait for threads
            for t in threads:
                t.join()

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--clients-per-round', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar100', choices=['Cifar10', 'MNIST', 'Cifar100'])
    parser.add_argument('--test-client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    server = FederatedLearningServer(args)
    server.run_server()

if __name__ == '__main__':
    main()