#!/usr/bin/env bash
# Stop TensorFold server
# by Screwed Up Tech — screwedup.tech

echo "[*] Stopping TensorFold..."
pkill -9 -f "tensorfold serve" 2>/dev/null && echo "[*] Stopped." || echo "[*] Not running."
