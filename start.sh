#!/bin/bash

# Ensure PORT is set, default to 8000 if not
PORT=${PORT:-8000}

# Start FastAPI with Uvicorn
uvicorn main:app --host 0.0.0.0 --port=$PORT
