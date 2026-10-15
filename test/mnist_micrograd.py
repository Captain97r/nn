#!/usr/bin/env python3
"""Micrograd (well-known project) side of the MNIST task.

Default mode (after the asm full run has written build/model.bin):
  1. cross-validation: bit-compare 20 test-image outputs between the trained
     asm net (called through build/neural.dll via ctypes) and the same
     weights fed through MGNet (micrograd.Values, identical op order);
  2. recognition timing: forward-only, asm DLL vs micrograd.

--bench mode (no trained model needed):
  time micrograd forward+backward per sample and the 269322-param update with
  FRESH He weights (seed 20261014), extrapolate to the asm 60k x 12 run.

Usage:
  python test/mnist_micrograd.py
  python test/mnist_micrograd.py --bench
"""
import ctypes
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "micrograd"))
sys.path.insert(0, HERE)
sys.setrecursionlimit(20000)  # micrograd build_topo recurses ~1400 deep on 784-wide nets

from micrograd.engine import Value  # noqa: E402,F401
from test_micrograd import MGNet, he_init  # noqa: E402

SIZES = [784, 256, 256, 10]
L = len(SIZES) - 1
WTOT = 256 * 784 + 256 * 256 + 10 * 256
BTOT = 256 + 256 + 10
SEED = 20261014
N_TRAIN, EPOCHS = 60000, 12


def load_model(path):
    """(sizes, flat params) in the model.bin layout written by train_mnist.c:
    int32 L, int32 res, int32 wtot, int32 btot, int32 sizes[8], W, B."""
    with open(path, "rb") as f:
        lh, res, wtot, btot = struct.unpack("<iiii", f.read(16))
        sizes = struct.unpack("<8i", f.read(32))
        W = struct.unpack("<%dd" % wtot, f.read(8 * wtot))
        B = struct.unpack("<%dd" % btot, f.read(8 * btot))
    assert (lh, res, wtot, btot) == (L, 0, WTOT, BTOT)
    return list(sizes[:L + 1]), list(W) + list(B)


def load_test(n):
    data = os.path.join(ROOT, "data")
    with open(os.path.join(data, "mnist_test_imgs.bin"), "rb") as f:
        raw = f.read(784 * n)
    with open(os.path.join(data, "mnist_test_labels.bin"), "rb") as f:
        T = list(f.read(n))
    return [b / 255.0 for b in raw], T


def load_dll():
    lib = ctypes.CDLL(os.path.join(ROOT, "build", "neural.dll"))
    lib.nn_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                              ctypes.c_uint64, ctypes.POINTER(ctypes.c_double)]
    lib.nn_create.restype = ctypes.c_void_p
    lib.nn_free.argtypes = [ctypes.c_void_p]
    lib.nn_set_weights.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double)]
    lib.nn_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double)]
    return lib


def make_net(lib, params):
    sizes_arr = (ctypes.c_int * 8)()
    for _i, _s in enumerate(SIZES):
        sizes_arr[_i] = _s
    lr = ctypes.c_double(0.1)
    net = lib.nn_create(L, sizes_arr, SEED, ctypes.byref(lr))
    flat = (ctypes.c_double * len(params))(*params)
    lib.nn_set_weights(net, flat,
                       (ctypes.c_double * BTOT)(*params[WTOT:]))
    return net


def asm_forward_many(lib, net, X, n):
    """n asm forwards; X is a flat list of doubles (already /255)."""
    xb = (ctypes.c_double * 784)()
    yb = (ctypes.c_double * 10)()
    out = []
    for k in range(n):
        xb[:] = X[k * 784:(k + 1) * 784]
        lib.nn_forward(net, xb, yb)
        out.append(list(yb))
    return out


def mg_forward(net, x):
    return [v.data for v in net.forward(x)]


def argmax(v):
    m, am = v[0], 0
    for i in range(1, len(v)):
        if v[i] > m:
            m, am = v[i], i
    return am


def cross_validate(model_path, n=20):
    print(f"--- cross-validation: asm DLL vs micrograd, {n} test images ---")
    sizes, params = load_model(model_path)
    X, T = load_test(n)
    lib = load_dll()
    net = make_net(lib, params)
    y_asm = asm_forward_many(lib, net, X, n)
    lib.nn_free(net)
    mg = MGNet.from_flat(sizes, params)
    exact = agree = 0
    maxdiff = 0.0
    for k in range(n):
        x = X[k * 784:(k + 1) * 784]
        y_mg = mg_forward(mg, x)
        d = max(abs(a - b) for a, b in zip(y_asm[k], y_mg))
        maxdiff = max(maxdiff, d)
        exact += all(a == b for a, b in zip(y_asm[k], y_mg))
        agree += argmax(y_asm[k]) == argmax(y_mg)
        ta, tm = argmax(y_asm[k]), argmax(y_mg)
        mark = "ok" if (argmax(y_asm[k]) == T[k]) else "MISS"
        print(f"  img {k:2d}: true={T[k]} asm={ta} mg={tm} "
              f"max|diff|={d:.3g} {mark}")
    print(f"  bit-exact outputs: {exact}/{n}, argmax agreement: {agree}/{n}, "
          f"max |diff| over all {n*10} outputs: {maxdiff:.3g}")
    return params


def recog_timing(params, n=50):
    print(f"--- recognition (forward only), {n} test images ---")
    X, _ = load_test(n)
    lib = load_dll()
    net = make_net(lib, params)
    t0 = time.perf_counter()
    asm_forward_many(lib, net, X, n)
    t_asm = (time.perf_counter() - t0) / n
    lib.nn_free(net)
    mg = MGNet.from_flat(list(SIZES), params)
    t0 = time.perf_counter()
    for k in range(n):
        mg_forward(mg, X[k * 784:(k + 1) * 784])
    t_mg = (time.perf_counter() - t0) / n
    print(f"  asm       {t_asm*1e6:9.1f} us/image")
    print(f"  micrograd {t_mg*1e6:9.1f} us/image")
    print(f"  speedup   {t_mg / t_asm:9.0f}x")


def read_asm_epoch():
    """Average per-epoch seconds from the asm full-run log, if present."""
    log = os.path.join(ROOT, "build", "full_run.log")
    if not os.path.exists(log):
        return None
    es = []
    for line in open(log):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "epoch" and parts[2].endswith("ms"):
            es.append(float(parts[2][:-2]) / 1000.0)
    return sum(es) / len(es) if es else None


def bench(n_samples=5):
    print("--- micrograd training speed (fresh He weights, seed 20261014) ---")
    W, B = he_init(SIZES, SEED)
    net = MGNet(SIZES, W, B)
    # one training sample per digit, from the first 600 training samples
    data = os.path.join(ROOT, "data")
    with open(os.path.join(data, "mnist_train_imgs.bin"), "rb") as f:
        raw = f.read(784 * 600)
    with open(os.path.join(data, "mnist_train_labels.bin"), "rb") as f:
        T = list(f.read(600))
    X = [b / 255.0 for b in raw]
    picked = [k for k in range(600) if T[k] == 1]  # 5 distinct samples
    ts = []
    for k in picked:
        x = X[k * 784:(k + 1) * 784]
        t = [1.0 if j == T[k] else 0.0 for j in range(10)]
        t0 = time.perf_counter()
        net.loss(x, t).backward()
        ts.append(time.perf_counter() - t0)
    t_fb = sum(ts) / len(ts)
    t0 = time.perf_counter()
    for p in net.params():
        p.data -= 0.1 * p.grad
    t_up = time.perf_counter() - t0
    epoch_mg = N_TRAIN * t_fb + t_up
    print(f"  micrograd fwd+bwd per sample:  {t_fb*1000:10.1f} ms")
    print(f"  micrograd update ({WTOT+BTOT} params): {t_up*1000:8.1f} ms")
    print(f"  micrograd 1 epoch (60000 samples): ~{epoch_mg:8.0f} s")
    print(f"  micrograd {EPOCHS} epochs: ~{EPOCHS*epoch_mg/3600:6.1f} h")
    asm_ep = read_asm_epoch()
    if asm_ep:
        print(f"  asm (measured, full_run.log): {asm_ep:6.1f} s/epoch, "
              f"{EPOCHS*asm_ep/60:5.1f} min total -> speedup {epoch_mg/asm_ep:,.0f}x")
    return t_fb


def main():
    if "--bench" in sys.argv:
        bench()
        return 0
    model = os.path.join(ROOT, "build", "model.bin")
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    params = cross_validate(model)
    recog_timing(params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
