#!/usr/bin/env bash
# Downloads the ArcFace model into app/src/main/assets/models/.
# Run by CI before the build; you can also run it locally.
# NOTE: InsightFace pretrained models are for non-commercial / research use only.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/app/src/main/assets/models"
mkdir -p "$DEST"
OUT="$DEST/w600k_mbf.onnx"
MIN=12000000

if [ -s "$OUT" ] && [ "$(stat -c%s "$OUT")" -ge "$MIN" ]; then echo "have w600k_mbf.onnx"; exit 0; fi

TMP="$(mktemp -d)"
echo "downloading buffalo_s.zip (contains w600k_mbf.onnx)"
curl -fsSL --retry 3 -o "$TMP/buffalo_s.zip" \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip
unzip -joq "$TMP/buffalo_s.zip" w600k_mbf.onnx -d "$DEST"
rm -rf "$TMP"
[ "$(stat -c%s "$OUT")" -ge "$MIN" ] || { echo "ERROR: w600k_mbf.onnx too small, download failed" >&2; exit 1; }
ls -la "$DEST"
