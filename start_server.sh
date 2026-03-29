#!/bin/sh
exec uvicorn test.main:app --host 0.0.0.0 --port "$PORT"
