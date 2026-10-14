#!/usr/bin/env python3
"""Download MNIST (IDX format, gzipped) from the official cvdf-datasets
mirror (Yann LeCun's original data, standard for the task) and convert to
the raw binary files the C training driver consumes:

  data/mnist_train_imgs.bin    60000 * 784 bytes  (uint8, 0..255, row-major)
  data/mnist_train_labels.bin  60000 bytes        (digit per image)
  data/mnist_test_imgs.bin     10000 * 784 bytes
  data/mnist_test_labels.bin   10000 bytes

Idempotent: skips any file that already exists.
"""
import gzip
import os
import struct
import urllib.request

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

IMAGES = [
    ("train-images-idx3-ubyte.gz", "train_images.gz", "mnist_train_imgs.bin", 60000),
    ("t10k-images-idx3-ubyte.gz", "t10k_images.gz", "mnist_test_imgs.bin", 10000),
]
LABELS = [
    ("train-labels-idx1-ubyte.gz", "train_labels.gz", "mnist_train_labels.bin", 60000),
    ("t10k-labels-idx1-ubyte.gz", "t10k_labels.gz", "mnist_test_labels.bin", 10000),
]


def fetch(remote, local):
    if os.path.exists(local):
        return
    print(f"downloading {remote} ...", flush=True)
    urllib.request.urlretrieve(BASE + remote, local)
    print(f"  {os.path.getsize(local) / 1e6:.1f} MB", flush=True)


def main():
    os.makedirs(DATA, exist_ok=True)
    for remote, gz_name, out_name, n in IMAGES + LABELS:
        fetch(remote, os.path.join(DATA, gz_name))

    for _, gz_name, out_name, n in IMAGES:
        out = os.path.join(DATA, out_name)
        if os.path.exists(out):
            continue
        with gzip.open(os.path.join(DATA, gz_name)) as f:
            magic, cnt, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"bad image magic {magic:#x}"
            assert cnt == n, f"expected {n} images, got {cnt}"
            raw = f.read()
            assert len(raw) == cnt * rows * cols
        with open(out, "wb") as o:
            o.write(raw)
        print(f"{out_name}: {cnt} images {rows}x{cols}", flush=True)

    for _, gz_name, out_name, n in LABELS:
        out = os.path.join(DATA, out_name)
        if os.path.exists(out):
            continue
        with gzip.open(os.path.join(DATA, gz_name)) as f:
            magic, cnt = struct.unpack(">II", f.read(8))
            assert magic == 2049, f"bad label magic {magic:#x}"
            assert cnt == n, f"expected {n} labels, got {cnt}"
            raw = f.read()
        with open(out, "wb") as o:
            o.write(raw)
        print(f"{out_name}: {cnt} labels", flush=True)

    # sanity stats on the train pixels
    with open(os.path.join(DATA, "mnist_train_imgs.bin"), "rb") as f:
        head = f.read(200000)
    print(f"train pixel stats (first {len(head)} bytes): "
          f"min={min(head)} max={max(head)} mean={sum(head) / len(head):.2f} "
          f"nonzero={sum(1 for b in head if b) / len(head):.1%}")
    print("mnist data ready")


if __name__ == "__main__":
    main()
