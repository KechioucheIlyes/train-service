#!/bin/sh
set -e

echo "=== START TRAIN-SERVICE ==="
echo "Python version:"
python --version

echo "Launching training..."
python -m app.main

echo "=== TRAIN-SERVICE FINISHED ==="

if [ -n "$RUNPOD_POD_ID" ] && [ -n "$RUNPOD_API_KEY" ]; then
  echo "Stopping current Runpod pod: $RUNPOD_POD_ID"
  curl -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
    -H "Authorization: Bearer $RUNPOD_API_KEY"
else
  echo "RUNPOD_POD_ID or RUNPOD_API_KEY not found, skip self-stop."
fi