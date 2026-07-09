#!/bin/bash
# Shell entry point for HyperQO query optimization skill
# Usage: optimize_query.sh --dsn <dsn> --query <sql> [--config <json>]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/usr/local/python37/bin/python3.7"

exec "$PYTHON" "$SCRIPT_DIR/wrapper.py" "$@"
