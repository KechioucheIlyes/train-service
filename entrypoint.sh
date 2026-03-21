#!/bin/bash
set -e

echo "=== START TRAIN-SERVICE ==="
echo "Python version:"
python --version

echo "Launching training..."
python -m app.main

echo "=== TRAIN-SERVICE FINISHED ==="