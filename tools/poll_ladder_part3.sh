#!/bin/bash
# 30-min cadence v11 ladder poll, runs ~12h
export KAGGLE_API_TOKEN=$(grep "^KAGGLE_API_TOKEN=" .env | cut -d= -f2-)
for i in $(seq 1 24); do
  echo "=== poll $i at $(date +%H:%M:%S) ===" >> runs/v11_ladder_poll_part3_20260425.log
  .venv/Scripts/kaggle.exe competitions submissions -c orbit-wars 2>&1 | head -7 | tail -3 >> runs/v11_ladder_poll_part3_20260425.log
  sleep 1800
done
