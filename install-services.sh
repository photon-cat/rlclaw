#!/bin/bash
# Install all rlclaw services as systemd units (requires sudo)
# Usage: sudo ./install-services.sh

set -e
PROJ="/home/jacob/rlclaw"

echo "Stopping existing processes..."
pkill -f "tsx src/index" 2>/dev/null || true
pkill -f "tsx src/discord-bot" 2>/dev/null || true
pkill -f "tsx src/dashboard" 2>/dev/null || true
pkill -f "run-agent.sh" 2>/dev/null || true
sleep 2

echo "Installing systemd services..."
cp "$PROJ/rlclaw-agent.service" /etc/systemd/system/
cp "$PROJ/rlclaw-bot.service" /etc/systemd/system/
cp "$PROJ/rlclaw-dashboard.service" /etc/systemd/system/

systemctl daemon-reload

echo "Enabling and starting services..."
systemctl enable --now rlclaw-dashboard
systemctl enable --now rlclaw-bot
systemctl enable --now rlclaw-agent

echo ""
echo "All services installed and started."
echo "  systemctl status rlclaw-agent      # research agent (restarts on usage limit)"
echo "  systemctl status rlclaw-bot        # discord bot"
echo "  systemctl status rlclaw-dashboard  # web dashboard"
echo ""
echo "  journalctl -fu rlclaw-agent        # live agent logs"
echo "  journalctl -fu rlclaw-bot          # live bot logs"
echo ""
echo "All services start on boot automatically."
