import socket
import time
import argparse
import sys
import torch
import os
import struct
import pickle
import subprocess 

# Import internal modules
from ..utils.network_utils import send_data, recv_data
from ..utils.logger import setup_logger
from ..utils.model_utils import quantization, dequantization
from .trainer import Trainer

# Initialize logger
logger = setup_logger("client.main")

def apply_network_rules():
    """
    Applies Linux Traffic Control (tc) rules to limit bandwidth
    based on environment variables.
    """
    bw = os.environ.get('BANDWIDTH')
    label = os.environ.get('NETWORK_LABEL', 'Custom')
    
    if bw:
        try:
            logger.info(f"--- Applying Network Limit: {bw} ({label}) ---")
            # Clear existing rules (ignore error if none exist)
            subprocess.run("tc qdisc del dev eth0 root", shell=True, stderr=subprocess.DEVNULL)
            
            # Apply TBF (Token Bucket Filter)
            # rate: bandwidth
            # burst: buffer size (crucial for TCP, usually 10-32k for low speeds)
            # latency: max queue time before drop (400ms is generous for cellular)
            cmd = f"tc qdisc add dev eth0 root tbf rate {bw} burst 32kbit latency 400ms"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully limited outgoing bandwidth to {bw}")
            else:
                logger.error(f"Failed to apply network rule: {result.stderr}")
        except Exception as e:
            logger.error(f"Error applying network rules: {e}")
    else:
        logger.info("No bandwidth limit set (running at full speed).")

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--dataset', type=str, default='Cifar100', choices=['Cifar10', 'MNIST', 'FashionMNIST', 'Cifar100'])
    parser.add_argument('--client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument("--ala", type=int, default=1, help="1 to enable FedALA, 0 to disable")
    return parser.parse_args()

def main():
    args = parse_args()
    
    apply_network_rules()
    
    # Setup Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    logger.info(f"--- Starting FL Client {args.client_idx} ---")
    logger.info(f"Target Server: {args.host}:{args.port}")

    # 1. Initialize Trainer
    try:
        trainer = Trainer(args, args.client_idx)
        logger.info(f"Algorithm PUMA+FedALA initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        sys.exit(1)
        
    # Connection Loop
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    while not connected:
        try:
            sock.connect((args.host, args.port))
            connected = True
            logger.info(f"Connected to server {args.host}:{args.port}")
            send_data(sock, args.client_idx)
            server_status = recv_data(sock)
            logger.info(f"Successfully registered. Server status: {server_status}")
        except ConnectionRefusedError:
            logger.warning("Connection refused. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            sys.exit(1)

    # Main Communication Loop
    try:
        round_num = 0
        while True:
            logger.info(f"Waiting for model (Round {round_num})...")
            
            # --- RECEIVE PHASE ---
            start_download = time.time()
            payload = recv_data(sock)
            
            if payload == 'shutdown':
                logger.info("Server sent shutdown signal. Exiting.")
                break
            
            if payload is None:
                logger.error("Received None payload. Server might have closed connection.")
                break

            global_model_quant, prune_rate = payload
            
            download_time = time.time() - start_download
            payload_size_kb = sys.getsizeof(pickle.dumps(payload)) / 1024
            speed = payload_size_kb / download_time if download_time > 0 else 0
            
            logger.info(f"Downloaded model for round {round_num}: {payload_size_kb:.1f}KB in {download_time:.2f}s ({speed:.1f} KB/s)")
            logger.info(f"--- Starting Round {round_num} ---")
            
            if prune_rate > 0:
                logger.info(f"Received PRUNED model (ratio={prune_rate:.2f})")

            # Dequantize
            global_model_state = dequantization(global_model_quant)

            # --- TRAINING PHASE ---
            logger.info("Starting local training...")
            train_start = time.time()
            
            local_state, num_samples = trainer.train(
                global_state_dict=global_model_state,
                prune_rate=prune_rate,
                round_num=round_num,
                use_ala=(args.ala == 1)
            )
            
            train_time = time.time() - train_start
            logger.info(f"Local training complete in {train_time:.2f}s.")

            # --- SEND PHASE ---
            logger.info(f"Submitting trained model update from {num_samples} samples...")
            quantized_local_state = quantization(local_state)
            
            metrics = {
                'samples': num_samples,
                'training_time': train_time,
                'download_time': download_time,
                'client_id': args.client_idx
            }
            
            upload_start = time.time()
            send_data(sock, (quantized_local_state, metrics))
            upload_time = time.time() - upload_start
            
            logger.info(f"Successfully submitted update in {upload_time + train_time + download_time:.2f}s")
            logger.info("Update submitted. Waiting for next round...")
            
            round_num += 1

    except KeyboardInterrupt:
        logger.info("Client stopping...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        logger.info("Connection closed.")

if __name__ == '__main__':
    main()