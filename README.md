# FaceLock

Android (Kotlin) face-lock app, built by GitHub Actions.

## How it works
- **Train / Enroll** — add faces two ways: *Capture live* (front camera) or *From gallery* (pick photos).
  Only embeddings are stored (in the app's private storage), no photos.
- **Unlock** — *live camera only*; there is no photo option on this screen. Optional blink check
  (open → closed → open) before the match is accepted.
- **Models** — MobileFaceNet (192-d), FaceNet-128, FaceNet-512. Settings screen: *Auto* (benchmark winner)
  or pick one manually, so you can try them side by side. Every enrolled face is embedded with all models,
  so switching the model needs no re-enrolment.
- **Matching** — cosine similarity vs. enrolled samples (mean of best 3), compared with the model's threshold
  plus the Strictness slider.

## Build (GitHub)
1. Push this folder to a GitHub repo (branch `main`).
2. Actions → *Build FaceLock APK* runs automatically:
   1. `benchmark` — downloads the models, tests all of them on LFW pairs, picks the most accurate,
      tunes thresholds and writes them into `models.json` (results appear in the run summary).
   2. `build` — compiles the APK with that winner as the *Auto* model.
3. Download the `FaceLock-debug-apk` artifact, install on your phone.

Manual run: Actions → *Run workflow* (you can untick the benchmark to skip it).

## Benchmark on your own photos (recommended)
LFW is a generic dataset; accuracy on *your* phone camera is what matters.
Put photos in folders by person (`people/ali/1.jpg`, `people/sara/1.jpg`, ... at least 2 photos each for 2+ people), then:

    pip install ai-edge-litert numpy pillow scikit-learn
    bash tools/fetch_models.sh
    python tools/benchmark.py --custom people --update-config

## Limits (be aware)
- The blink check stops a simple held-up photo, but **not** a video/mask replay. For a real security lock add a
  dedicated anti-spoof model.
- Don't use this to protect anything high-value without that.
- Model files are downloaded from third-party GitHub repos; check their licences before publishing the app.
# FaceLock
