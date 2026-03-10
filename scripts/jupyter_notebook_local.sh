#!/bin/bash
cd ~/personal/github_repos/claude-scratch-work/

echo

tmux new -s jupyternotebook -d "echo; echo; echo; PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter notebook --no-browser --port=7007 --ServerApp.max_body_size=1073741824 --ServerApp.max_buffer_size=1073741824"

echo
echo

sleep 15

echo
echo

tmux capture-pane -p -J -S -100000 -t jupyternotebook

