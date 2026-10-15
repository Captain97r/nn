#!/usr/bin/env python3
"""Throwaway: per-digit accuracy + confusion rows for the asm MNIST model."""
import ctypes
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [784, 256, 256, 10]
L = len(SIZES) - 1


def load_model(path):
    lib = ctypes.CDLL(os.path.join(HERE, "build", "neural.dll"))
    lib.nn_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                              ctypes.c_uint64, ctypes.POINTER(ctypes.c_double)]
    lib.nn_create.restype = ctypes.c_void_p
    lib.nn_set_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double)]
    lib.nn_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double)]
    with open(path, "rb") as f:
        data = f.read()
    lh, res, wtot, btot = struct.unpack_from("<iiii", data, 0)
    assert (lh, res) == (L, 0), "bad model header"
    sizes_arr = (ctypes.c_int * 8)()
    for i, s in enumerate(SIZES):
        sizes_arr[i] = s
    lr = ctypes.c_double(0.1)
    net = lib.nn_create(L, sizes_arr, 0, ctypes.byref(lr))
    assert net, "nn_create failed"
    params = (ctypes.c_double * (wtot + btot)).from_buffer_copy(
        data[16 + 32: 16 + 32 + (wtot + btot) * 8])
    lib.nn_set_weights(net, params,
                        ctypes.cast(ctypes.addressof(params) + wtot * 8,
                                    ctypes.POINTER(ctypes.c_double)))
    out = (ctypes.c_double * SIZES[-1])()

    def predict(img):
        lib.nn_forward(net, img, out)
        return list(out)

    return predict


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build", "model.bin")
    with open(os.path.join(HERE, "data", "mnist_train_labels.bin"), "rb") as f:
        tr = f.read()
    with open(os.path.join(HERE, "data", "mnist_test_labels.bin"), "rb") as f:
        te = f.read()
    with open(os.path.join(HERE, "data", "mnist_test_imgs.bin"), "rb") as f:
        img_data = f.read()
    n_te = len(te)

    trc = [0] * 10
    for b in tr:
        trc[b] += 1
    tec = [0] * 10
    for b in te:
        tec[b] += 1
    print("train counts:", dict(enumerate(trc)))
    print("test  counts:", dict(enumerate(tec)))

    predict = load_model(model)
    conf = [[0] * 10 for _ in range(10)]
    for k in range(n_te):
        raw = img_data[k * 784:(k + 1) * 784]
        img = (ctypes.c_double * 784)(*[b / 255.0 for b in raw])
        y = predict(img)
        pred = max(range(10), key=lambda j: y[j])
        conf[te[k]][pred] += 1

    print("\nper-digit test accuracy (model %s):" % os.path.basename(model))
    total = 0
    per = {}
    for d in range(10):
        per[d] = conf[d][d]
        total += conf[d][d]
    for d in range(10):
        print("  digit %d: %3d/%3d (%5.2f%%)" % (d, conf[d][d], tec[d],
                                                 100.0 * conf[d][d] / tec[d]))
    print("  overall: %d/%d (%.2f%%)" % (total, n_te, 100.0 * total / n_te))

    print("\nconfusion rows (rows=true, cols=predicted):")
    for d in range(10):
        row = "".join("%4d" % conf[d][p] for p in range(10))
        print("  true=%d: %s" % (d, row))

    zero_errs = [(d, n) for d, n in enumerate(conf[0]) if d != 0]
    print("\nzero mispredictions:", zero_errs)


if __name__ == "__main__":
    main()
