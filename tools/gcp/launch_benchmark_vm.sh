#!/usr/bin/env bash
# Create a self-deleting Spot V100 VM for soundersim GPU benchmarking.
# Cost: ~$0.82/h Spot (n1-standard-8 + 1x V100, us-central1).
# Safety: Spot + --max-run-duration + DELETE => hard spend cap even if
# forgotten; boot disk auto-deletes with the instance.
#
# Image note: plain Ubuntu 22.04, NOT the deeplearning images -- those bake
# driver 580 which dropped Volta (V100) support. The startup script installs
# the 550-server branch (supports V100, driver >= 525 satisfies jax[cuda12]
# pip CUDA) via DKMS and reboots once to load it.
set -euo pipefail
NAME=${1:-soundersim-v100}
STARTUP='#!/bin/bash
set -e
if [ ! -f /var/lib/soundersim-driver-done ]; then
  apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    nvidia-driver-550-server linux-headers-$(uname -r)
  touch /var/lib/soundersim-driver-done
  reboot
fi'
for ZONE in us-central1-a us-central1-c us-central1-f us-central1-b; do
  echo "trying $ZONE..."
  gcloud compute instances create "$NAME" \
    --zone=$ZONE --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --provisioning-model=SPOT --instance-termination-action=DELETE \
    --max-run-duration=6h \
    --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
    --boot-disk-size=80GB --boot-disk-type=pd-balanced \
    --scopes=cloud-platform \
    --metadata=startup-script="$STARTUP" && { echo "created in $ZONE"; exit 0; }
done
echo "no Spot V100 capacity in us-central1" >&2; exit 1
