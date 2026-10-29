# FaceLock (v0.2)

Android (Kotlin) face-lock app, built by GitHub Actions.

## How it works
- **Detection / alignment** — Google ML Kit finds the face and its landmarks (eyes, mouth). The face is warped to the
  standard 112x112 ArcFace pose (similarity transform) before recognition. ML Kit itself does *not* recognise faces.
- **Recognition** — InsightFace `w600k_mbf` (ArcFace MobileFaceNet, 512-d) via ONNX Runtime. Only embeddings are stored.
- **Enroll** — live capture (recommended) or gallery. Only near-frontal, big-enough faces are accepted.
  Add 6+ live samples: straight, slight angles, with and without glasses.
- **Unlock** — live camera only. Optional blink check, then 5 consecutive frames must average above
  `threshold (0.50) + strictness slider` and none may be far below it.

## Build (GitHub)
1. Push this folder to a GitHub repo (branch `main`).
2. Actions → *Build FaceLock APK* downloads the model and compiles the APK.
3. Download the `FaceLock-debug-apk` artifact and install it.

## Tuning
The Unlock screen shows the score. Note your own score and a look-alike's score, then set the strictness slider
between them (closer to your own lower values than to the look-alike's highest). Re-enroll after upgrading from v0.1
("Delete all enrolled faces" first) because the model changed.

## Limits
- No anti-spoofing: a video/mask replay can fool a plain 2D camera. Don't protect anything high-value with this.
- InsightFace pretrained models are for non-commercial / research use only.
