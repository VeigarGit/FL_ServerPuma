#!/bin/bash

# Create tmux session
tmux new-session -d -s myapp 'python server.py'

# Create panes for clients
tmux split-window -h 'python client.py --client-idx 0'
tmux split-window -v 'python client.py --client-idx 1'

