#!/usr/bin/env bash
# Downloads the face-embedding models into app/src/main/assets/models/.
# Run by CI before the benchmark and the build; you can also run it locally.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/app/src/main/assets/models"
mkdir -p "$DEST"

FACENET=https://raw.githubusercontent.com/shubham0204/FaceRecognition_With_FaceNet_Android/master/app/src/main/assets
MFN=https://raw.githubusercontent.com/syaringan357/Android-MobileFaceNet-MTCNN-FaceAntiSpoofing/master/app/src/main/assets

fetch() { # url, filename, minimum size in bytes
  local url="$1" out="$DEST/$2" min="$3"
  if [ -s "$out" ] && [ "$(stat -c%s "$out")" -ge "$min" ]; then echo "have $2"; return; fi
  echo "downloading $2"
  curl -fsSL --retry 3 -o "$out" "$url"
  [ "$(stat -c%s "$out")" -ge "$min" ] || { echo "ERROR: $2 is too small, download failed" >&2; exit 1; }
}

fetch "$MFN/MobileFaceNet.tflite"       mobilefacenet.tflite 4000000
fetch "$FACENET/facenet.tflite"         facenet.tflite       20000000
fetch "$FACENET/facenet_512.tflite"     facenet_512.tflite   20000000

ls -la "$DEST"
