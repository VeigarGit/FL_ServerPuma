#!/bin/bash

# ==============================================================================
# PUMA FL Simulation Runner
# ==============================================================================

# 1. Detect Docker Compose Command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo -e "\033[0;31mError: Docker Compose not found. Please install it or check your PATH.\033[0m"
    exit 1
fi

# Default Configuration
COMPOSE_FILE="docker-compose.generated.yml"
GENERATE_SCRIPT="generate_compose.py"
# Create a unique log file name based on time
LOG_FILE="simulation_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PUMA Federated Learning Simulation Manager ===${NC}"
echo -e "${YELLOW}Using command: ${DOCKER_COMPOSE_CMD}${NC}"

# ------------------------------------------------------------------------------
# 2. Cleanup Function (Runs on Exit)
# ------------------------------------------------------------------------------
cleanup() {
    echo -e "\n${YELLOW}Shutting down simulation...${NC}"
    if [ -f "$COMPOSE_FILE" ]; then
        $DOCKER_COMPOSE_CMD -f "$COMPOSE_FILE" down --remove-orphans
    fi
    echo -e "${GREEN}Cleanup complete. Logs saved to: ${LOG_FILE}${NC}"
}

# Trap Ctrl+C (SIGINT) and Exit signals to ensure cleanup happens
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# 3. Build Step (Optional via flag, or if images missing)
# ------------------------------------------------------------------------------
# Check if images exist
if [[ "$(docker images -q fl_server 2> /dev/null)" == "" ]] || [[ "$1" == "--rebuild" ]]; then
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    echo -e "Building Server..."
    docker build -t fl_server -f dockerfile.server .
    echo -e "Building Client..."
    docker build -t fl_client -f dockerfile.client .
else
    echo -e "${GREEN}Docker images found. Skipping build (use --rebuild to force).${NC}"
fi

# ------------------------------------------------------------------------------
# 4. Generate Configuration
# ------------------------------------------------------------------------------
echo -e "${YELLOW}Generating Docker Compose configuration...${NC}"

if [ -f "$GENERATE_SCRIPT" ]; then
    python3 "$GENERATE_SCRIPT"
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "${RED}Error: Failed to generate $COMPOSE_FILE${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: Generation script '$GENERATE_SCRIPT' not found.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 5. Run Simulation
# ------------------------------------------------------------------------------
echo -e "${GREEN}Starting Simulation...${NC}"
echo -e "${GREEN}Logs will be saved to: ${LOG_FILE}${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop.${NC}"
echo -e "---------------------------------------------------------"

# Run with pipe to 'tee' to save logs to file AND show in console
# 2>&1 redirects errors to the same stream so we capture crashes too
$DOCKER_COMPOSE_CMD -f "$COMPOSE_FILE" up --abort-on-container-exit --force-recreate --remove-orphans 2>&1 | tee "$LOG_FILE"
