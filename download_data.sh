#!/usr/bin/env bash
set -euo pipefail

# ========= Config (override via env or edit here) =========
DOWNLOAD_MODE="${DOWNLOAD_MODE:-shards}"   # "single" or "shards"
OUT_DIR="${OUT_DIR:-$PWD/data}"            # where to save files
SINGLE_URL="${SINGLE_URL:-https://storage.googleapis.com/searchless_chess/data/puzzles.csv}"

# Shard settings
NUM_SHARDS=2148
BASE_URL="https://storage.googleapis.com/searchless_chess/data/train"
# ==========================================================

# Ensure aria2c is available (Debian/Ubuntu: sudo apt-get install -y aria2)
if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c not found. Install it first, e.g.:"
  echo "  sudo apt-get update && sudo apt-get install -y aria2"
  echo "On Arch: sudo pacman -S aria2   On Fedora: sudo dnf install aria2"
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ "$DOWNLOAD_MODE" == "single" ]]; then
  echo "Downloading single file: $SINGLE_URL"
  aria2c -c -x 16 -s 16 -d "$OUT_DIR" "$SINGLE_URL"

elif [[ "$DOWNLOAD_MODE" == "shards" ]]; then
  echo "Downloading action_value shards (00000..$(printf '%05d' $((NUM_SHARDS-1)))) ..."
  TMP_LIST="$(mktemp)"
  trap 'rm -f "$TMP_LIST"' EXIT

  # Build URL list (faster than invoking aria2 per file)
  # Files are named: action_value-00000-of-02148_data.bag ... action_value-02147-of-02148_data.bag
  last=$((NUM_SHARDS - 1))
  for i in $(seq -w 00000 "$last"); do
    echo "${BASE_URL}/action_value-${i}-of-$(printf '%05d' "$NUM_SHARDS")_data.bag" >> "$TMP_LIST"
  done

  # Batch download with resume, 16 connections, save to OUT_DIR
  aria2c -c -x 16 -s 16 -d "$OUT_DIR" -i "$TMP_LIST"

else
  echo "Unknown DOWNLOAD_MODE: $DOWNLOAD_MODE (expected 'single' or 'shards')" >&2
  exit 1
fi

echo "Done. Files are in: $OUT_DIR"
