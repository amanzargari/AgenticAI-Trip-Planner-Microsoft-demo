#!/usr/bin/env bash
# deploy.sh — pull latest code and restart all containers on the server.
# Usage:
#   ./deploy.sh           — full rebuild + restart
#   ./deploy.sh --no-build — restart without rebuilding images
set -euo pipefail

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

echo "==> Pulling latest code..."
git pull --ff-only

if [[ "${1:-}" != "--no-build" ]]; then
    echo "==> Building images (parallel)..."
    $COMPOSE build --parallel
fi

echo "==> Starting / restarting services..."
$COMPOSE up -d --remove-orphans

echo "==> Waiting for web-ui to respond..."
for i in $(seq 1 30); do
    if curl -sf http://localhost/health > /dev/null 2>&1; then
        echo "==> App is up."
        break
    fi
    echo "    attempt $i/30..."
    sleep 3
done

echo ""
echo "==> Container status:"
$COMPOSE ps

echo ""
echo "App running at: http://$(curl -sf ifconfig.me 2>/dev/null || echo '<server-ip>')"
