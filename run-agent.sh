#!/bin/bash
# Auto-restarting agent wrapper
# Restarts after 10 min on crash (handles usage limit resets)

PROJ="/home/jacob/rlclaw"
cd "$PROJ"

while true; do
  echo "[$(date)] Starting agent..."
  CLAUDECODE="" npx tsx src/index.ts >> /tmp/rlclaw-session.log 2>&1
  EXIT_CODE=$?
  echo "[$(date)] Agent exited with code $EXIT_CODE. Restarting in 10 minutes..."
  sleep 600
done
