#!/usr/bin/env bash

# .env loading
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo ".env file not found"
    exit 1
fi

exec "$@"