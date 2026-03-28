#!/usr/bin/env bash
# Example Verda "Startup script" — runs once at first boot as root (typical cloud-init).
# Customize: image pull, git clone, GPU driver packages if Verda does not preinstall them.
set -euo pipefail

LOG=/var/log/verda-bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "=== verda-bootstrap start $(date -Iseconds) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl gnupg

# Docker Engine (official convenience script — replace with distro packages if your policy forbids)
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi

systemctl enable docker
systemctl start docker

# Optional: add user to docker group if you SSH as non-root (uncomment and set USER)
# usermod -aG docker ubuntu

echo "=== verda-bootstrap done $(date -Iseconds) ==="
