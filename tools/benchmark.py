#!/usr/bin/env python3
"""
Benchmarks every model in app/src/main/assets/models.json on face-verification pairs and
picks the most accurate one.

  * Data: LFW pairs (scikit-learn downloads it) or your own photos (--custom DIR, one sub-folder per person).
  * For each model: the threshold is tuned on one split and accuracy is reported on a separate split.
  * --update-config writes the winner as "default" and the tuned thresholds into models.json,
    so the app's "Auto" mode uses the best model.

Preprocessing here mirrors the Kotlin code (Embedder.kt): square crop -> resize -> normalise -> L2.
NOTE: LFW images are pre-centred, so a centre crop stands in for the app's face detector. Numbers are a
fair *relative* comparison of the models, but for the real answer also run with --custom on photos taken
with your own phone.
"""
import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # fall back to full tensorflow if that is what is installed
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app/src/main/assets"


class Model:
    def __init__(self, spec):
        self.spec = spec
        self.it = Interpreter(model_path=str(ASSETS / "models" / spec["file"]))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.batch, self.size = int(self.inp["shape"][0]), int(self.inp["shape"][1])

    def embed(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32)
        if self.spec.get("norm") == "per_image":
            std = max(a.std(), 1.0 / np.sqrt(a.size))
            a = (a - a.mean()) / std
        else:
            a = (a - self.spec.get("mean", 127.5)) / self.spec.get("std", 128.0)
        x = np.repeat(a[None], self.batch, axis=0).astype(np.float32)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        v = self.it.get_tensor(self.out["index"])[0].astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-10)


def centre_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    s = int(min(w, h) * frac)
    l, t = (w - s) // 2, (h - s) // 2
    return img.crop((l, t, l + s, t + s))


def lfw_pairs(frac):
    from sklearn.datasets import fetch_lfw_pairs

    sets = {}
    for subset in ("train", "test"):
        d = fetch_lfw_pairs(subset=subset, color=True, slice_=None, resize=1.0, funneled=True)
        imgs = d.pairs  # (n, 2, H, W, 3) floats 0..255 (or 0..1 on old versions)
        scale = 255.0 if imgs.max() <= 1.0 else 1.0
        pairs = []
        for (a, b), y in zip(imgs, d.target):
            ia = Image.fromarray((a * scale).astype(np.uint8))
            ib = Image.fromarray((b * scale).astype(np.uint8))
            pairs.append((centre_crop(ia, frac), centre_crop(ib, frac), int(y)))
        sets[subset] = pairs
    return sets


def custom_pairs(folder, n_pairs=1500, seed=0):
    rnd = random.Random(seed)
    people = {}
    for p in sorted(Path(folder).iterdir()):
        if p.is_dir():
            files = [f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if files:
                people[p.name] = files
    multi = [k for k, v in people.items() if len(v) >= 2]
    if len(multi) < 2 or len(people) < 2:
        sys.exit("--custom needs >= 2 people, with >= 2 photos each for at least 2 of them")
    load = lambda f: Image.open(f)
    pos, neg = [], []
    for _ in range(n_pairs):
        k = rnd.choice(multi)
        a, b = rnd.sample(people[k], 2)
        pos.append((load(a), load(b), 1))
        k1, k2 = rnd.sample(list(people), 2)
        neg.append((load(rnd.choice(people[k1])), load(rnd.choice(people[k2])), 0))
    allp = pos + neg
    rnd.shuffle(allp)
    cut = len(allp) // 2
    return {"train": allp[:cut], "test": allp[cut:]}


def similarities(model, pairs):
    sims, labels = [], []
    for a, b, y in pairs:
        sims.append(float(np.dot(model.embed(a), model.embed(b))))
        labels.append(y)
    return np.array(sims), np.array(labels)


def best_threshold(sims, labels):
    best_t, best_acc = 0.0, 0.0
    for t in np.unique(np.round(sims, 3)):
        acc = float(((sims >= t) == labels.astype(bool)).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t, best_acc


def far_frr(sims, labels, t):
    same, diff = sims[labels == 1], sims[labels == 0]
    return float((diff >= t).mean()), float((same < t).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom", help="folder with one sub-folder per person (uses your own photos instead of LFW)")
    ap.add_argument("--crop", type=float, default=0.6, help="LFW centre-crop fraction (face share of the image)")
    ap.add_argument("--update-config", action="store_true", help="write winner + thresholds into models.json")
    ap.add_argument("--out", default=str(ROOT / "benchmark_results.json"))
    args = ap.parse_args()

    cfg_path = ASSETS / "models.json"
    cfg = json.loads(cfg_path.read_text())
    specs = [m for m in cfg["models"] if (ASSETS / "models" / m["file"]).exists()]
    if not specs:
        sys.exit("no model files found - run tools/fetch_models.sh first")

    sets = custom_pairs(args.custom) if args.custom else lfw_pairs(args.crop)
    print(f"pairs: train={len(sets['train'])} test={len(sets['test'])}")

    results = []
    for spec in specs:
        m = Model(spec)
        s_tr, y_tr = similarities(m, sets["train"])
        t, tr_acc = best_threshold(s_tr, y_tr)
        s_te, y_te = similarities(m, sets["test"])
        acc = float(((s_te >= t) == y_te.astype(bool)).mean())
        far, frr = far_frr(s_te, y_te, t)
        results.append(dict(id=spec["id"], label=spec["label"], threshold=round(t, 3),
                            train_acc=round(tr_acc, 4), test_acc=round(acc, 4),
                            far=round(far, 4), frr=round(frr, 4)))
        print(f"{spec['id']:14s} thr={t:.3f}  train={tr_acc:.4f}  TEST acc={acc:.4f}  FAR={far:.4f}  FRR={frr:.4f}")

    # best accuracy wins; ties -> lower false-accept rate (safer for a lock)
    results.sort(key=lambda r: (-r["test_acc"], r["far"]))
    winner = results[0]
    print(f"\nWINNER: {winner['id']} ({winner['test_acc']*100:.2f}% on held-out pairs)")
    Path(args.out).write_text(json.dumps(dict(winner=winner["id"], results=results,
                                              data="custom" if args.custom else "LFW"), indent=2))

    md = os.environ.get("GITHUB_STEP_SUMMARY")
    if md:
        with open(md, "a") as f:
            f.write("## Face model benchmark\n\n| model | threshold | test accuracy | FAR | FRR |\n|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['id']}{' 🏆' if r is winner else ''} | {r['threshold']} | {r['test_acc']*100:.2f}% | {r['far']*100:.2f}% | {r['frr']*100:.2f}% |\n")

    if args.update_config:
        by_id = {r["id"]: r for r in results}
        for m in cfg["models"]:
            if m["id"] in by_id:
                m["threshold"] = by_id[m["id"]]["threshold"]
        cfg["default"] = winner["id"]
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print("models.json updated")


if __name__ == "__main__":
    main()
