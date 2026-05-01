#!/bin/bash
# Better cadence + correct row range (head -8 | tail -5 = top 5 submissions)
export KAGGLE_API_TOKEN=$(grep "^KAGGLE_API_TOKEN=" .env | cut -d= -f2-)
LOG=runs/v11_ladder_poll_part4_20260425.log
for i in $(seq 1 24); do
  echo "=== poll $i at $(date +%H:%M:%S) ===" >> $LOG
  .venv/Scripts/kaggle.exe competitions submissions -c orbit-wars 2>&1 | head -8 | tail -5 >> $LOG
  sleep 1800
done
