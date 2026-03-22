#!/bin/sh
set -e

echo "=== START TRAIN-SERVICE ==="
echo "Python version:"
python --version

echo "Launching training..."
python -m app.main

echo "=== TRAIN-SERVICE FINISHED ==="

if [ -z "$RUNPOD_POD_ID" ]; then
  echo "RUNPOD_POD_ID not found, cannot stop pod."
  exit 1
fi

echo "Stopping current Runpod pod with runpodctl: $RUNPOD_POD_ID"
runpodctl stop pod "$RUNPOD_POD_ID"
