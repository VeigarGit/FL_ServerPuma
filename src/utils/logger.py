import logging
import sys

def setup_logger(name=__name__, level=logging.INFO):
    
    logger = logging.getLogger(name)
    
    # Check if the logger already has handlers to prevent duplicate logs
    if not logger.handlers:
        # Create a handler that writes to standard output (console)
        handler = logging.StreamHandler(sys.stdout)
        
        # Define log
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        
        logger.propagate = False

    return logger