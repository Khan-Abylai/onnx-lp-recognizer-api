#!/bin/bash
python3 -m uvicorn main:app --workers 8 --host 0.0.0.0 --port 9001