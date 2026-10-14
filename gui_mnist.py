#!/usr/bin/env python3
"""Tkinter GUI: browse the MNIST test set, live-predict with the pure-assembly
net (through build/neural.dll, ctypes), optionally cross-check the current
image with micrograd.

The asm net is only called from the main thread (its forward uses static
scratch buffers); "verify 1000" runs on the main thread too since 1000 asm
forwards take ~0.2 s. Micrograd (when enabled) runs in a worker thread
because one micrograd forward takes ~0.3 s.

Usage:
  python gui_mnist.py [--model build/model.bin]
  python gui_mnist.py --selftest [--model build/model.bin]
--selftest: headless; predicts the first 200 test images, prints accuracy,
exits. Use it to verify the GUI path without a display.
"""
import ctypes
import os
import random
import struct
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "test"))

SIZES = [784, 256, 256, 10]
L = len(SIZES) - 1
WTOT = 256 * 784 + 256 * 256 + 10 * 256
BTOT = 256 + 256 + 10
SEED = 20261014
STRIP = 24
CELL = 62
BIG = 336  # 28 px * 12


# ---------------- asm net via ctypes ----------------
def load_asm_net(model_path):
    lib = ctypes.CDLL(os.path.join(HERE, "build", "neural.dll"))
    lib.nn_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                              ctypes.c_uint64, ctypes.POINTER(ctypes.c_double)]
    lib.nn_create.restype = ctypes.c_void_p
    lib.nn_free.argtypes = [ctypes.c_void_p]
    lib.nn_set_weights.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double)]
    lib.nn_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double)]

    with open(model_path, "rb") as f:
        data = f.read()
    lh, res, wtot, btot = struct.unpack_from("<iiii", data, 0)
    assert (lh, res, wtot, btot) == (L, 0, WTOT, BTOT), "bad model"
    params = list(struct.unpack_from("<%dd" % (wtot + btot), data, 16 + 32))

    sizes_arr = (ctypes.c_int * 8)()
    for _i, _s in enumerate(SIZES):
        sizes_arr[_i] = _s
    lr = ctypes.c_double(0.1)
    net = lib.nn_create(L, sizes_arr, SEED, ctypes.byref(lr))
    flat = (ctypes.c_double * len(params))(*params)
    lib.nn_set_weights(net, flat,
                       (ctypes.c_double * BTOT)(*params[WTOT:]))
    xb = (ctypes.c_double * 784)()
    yb = (ctypes.c_double * 10)()

    def predict(xbytes):
        xb[:] = [b / 255.0 for b in xbytes]
        lib.nn_forward(net, xb, yb)
        return list(yb)

    return predict


# ---------------- test data ----------------
def load_test_set():
    data = os.path.join(HERE, "data")
    with open(os.path.join(data, "mnist_test_imgs.bin"), "rb") as f:
        imgs = f.read()
    with open(os.path.join(data, "mnist_test_labels.bin"), "rb") as f:
        T = f.read()
    return [imgs[i * 784:(i + 1) * 784] for i in range(len(imgs) // 784)], list(T)


def argmax(y):
    m, am = 0, 0
    for i in range(1, 10):
        if y[i] > m:
            m, am = y[i], i
    return am


def make_photo(px, scale=1):
    """px: 784 bytes, MNIST convention (255 = dark digit). White bg, black digit.
    One nested-list put: each inner list is one row of pixel colors (the only
    put() form that fills a full row; flat lists become 1-px-wide columns)."""
    rows = []
    for r in range(28):
        row = []
        for c in range(28):
            v = 255 - px[r * 28 + c]
            row.append(f"#{v:02x}{v:02x}{v:02x}")
        rows.append(row)
    ph = tk.PhotoImage(width=28, height=28)
    ph.put(rows)
    if scale != 1:
        ph = ph.zoom(scale, scale)
    return ph


# ---------------- micrograd (lazy, worker thread) ----------------
class MGBridge:
    def __init__(self, params_flat):
        sys.path.insert(0, os.path.join(HERE, "micrograd"))
        sys.setrecursionlimit(20000)
        from test_micrograd import MGNet
        self.net = MGNet.from_flat(list(SIZES), list(params_flat))

    def predict(self, xbytes):
        x = [b / 255.0 for b in xbytes]
        return [v.data for v in self.net.forward(x)]


# ---------------- selftest (headless) ----------------
def selftest(model_path, n=200):
    predict = load_asm_net(model_path)
    imgs, T = load_test_set()
    ok = 0
    t0 = time.perf_counter()
    for k in range(n):
        if argmax(predict(imgs[k])) == T[k]:
            ok += 1
    dt = (time.perf_counter() - t0) / n
    print(f"selftest: {ok}/{n} correct ({ok / n:.1%}), "
          f"{dt * 1e6:.1f} us/image (asm DLL)")
    for k in range(5):
        y = predict(imgs[k])
        am = argmax(y)
        print(f"  img {k}: true={T[k]} pred={am} conf={y[am]:.4f}")
    return 0 if ok / n > 0.5 else 1


# ---------------- GUI ----------------
class App:
    def __init__(self, root, model_path):
        self.root = root
        self.imgs, self.T = load_test_set()
        self.n = len(self.T)
        self.predict = load_asm_net(model_path)
        self.params_path = model_path
        self.mg = None
        self.mg_thread = None
        self.mg_line = None  # (pred, conf) for current idx
        self.idx = 0
        self.batch_pred = {}  # idx -> (pred, conf) after verify-1000
        self.photo_cache = {}  # (idx, scale) -> PhotoImage (held refs: Tk frees
                               # images whose Python wrapper is garbage-collected)

        root.title("MNIST — pure-assembly MLP recognizer")
        root.configure(bg="white")

        self.cv = tk.Canvas(root, width=1440, height=520, bg="white",
                            highlightthickness=0)
        self.cv.pack()

        bar = tk.Frame(root, bg="white")
        bar.pack(fill="x")
        tk.Button(bar, text="◀ prev", command=self.prev).pack(side="left", padx=2)
        self.slider = ttk.Scale(bar, from_=0, to=self.n - 1, orient="horizontal",
                                command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(bar, text="next ▶", command=self.next).pack(side="left", padx=2)
        tk.Button(bar, text="random", command=self.random).pack(side="left", padx=2)
        self.btn_verify = tk.Button(bar, text="✓ verify 1000",
                                    command=self.verify_1000)
        self.btn_verify.pack(side="left", padx=2)
        self.mg_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="micrograd", variable=self.mg_var,
                       command=self.on_mg_toggle).pack(side="left", padx=6)
        self.status = tk.Label(bar, text="", bg="white", fg="black")
        self.status.pack(side="left", padx=8)

        self.redraw()

    # ---- navigation ----
    def on_slider(self, v):
        self.go(int(float(v)))

    def random(self):
        self.go(random.randrange(self.n))

    def prev(self):
        self.go(max(0, self.idx - 1))

    def next(self):
        self.go(min(self.n - 1, self.idx + 1))

    def go(self, i):
        self.idx = i
        self.slider.set(i)
        self.redraw()

    def get_photo(self, i, scale):
        key = (i, scale)
        ph = self.photo_cache.get(key)
        if ph is None:
            ph = make_photo(self.imgs[i], scale)
            self.photo_cache[key] = ph
        return ph

    # ---- drawing ----
    def redraw(self):
        self.cv.delete("all")
        i = self.idx
        px = self.imgs[i]
        self.big_photo = self.get_photo(i, 12)
        self.cv.create_image(20, 20, image=self.big_photo, anchor="nw")
        y = self.predict(px)
        am = argmax(y)
        ok = am == self.T[i]
        color = "#0a0" if ok else "#c00"
        mark = "✓" if ok else "✗"
        self.cv.create_text(400, 60, anchor="w", font=("Consolas", 16),
                            text=f"img {i} / {self.n - 1}")
        self.cv.create_text(400, 100, anchor="w", font=("Consolas", 16),
                            text=f"true : {self.T[i]}")
        self.cv.create_text(400, 140, anchor="w", font=("Consolas", 16),
                            text=f"asm  : {am}  ({y[am]:.4f}) {mark}", fill=color)
        if self.mg_line is not None:
            pm, pmc = self.mg_line
            pok = "✓" if pm == self.T[i] else "✗"
            self.cv.create_text(400, 180, anchor="w", font=("Consolas", 16),
                                text=f"mg   : {pm}  ({pmc:.4f}) {pok}",
                                fill="#0a0" if pm == self.T[i] else "#c00")
        self.draw_strip()
        if (self.mg_var.get() and self.mg is not None
                and (self.mg_thread is None or not self.mg_thread.is_alive())):
            self.mg_current()

    def draw_strip(self):
        start = (self.idx // STRIP) * STRIP
        for k in range(STRIP):
            j = start + k
            if j >= self.n:
                break
            x0 = 20 + k * (CELL + 4)
            y0 = 400
            px = self.imgs[j]
            tph = self.get_photo(j, 2)
            self.cv.create_image(x0, y0, image=tph, anchor="nw")
            if j == self.idx:
                pred = argmax(self.predict(px))
                ok = pred == self.T[j]
            elif j in self.batch_pred:
                pred, _ = self.batch_pred[j]
                ok = pred == self.T[j]
            else:
                pred, ok = None, None
            outline = "#000" if j == self.idx else (
                "#0a0" if ok else ("#c00" if ok is False else "#bbb"))
            self.cv.create_rectangle(x0 - 2, y0 - 2, x0 + 58, y0 + 58,
                                     outline=outline,
                                     width=3 if j == self.idx else 1)
            self.cv.create_text(x0 + 28, y0 + 66,
                                text=str(pred) if pred is not None else "?",
                                font=("Consolas", 10),
                                fill=outline if ok is not None else "#888")

    # ---- micrograd ----
    def on_mg_toggle(self):
        if not self.mg_var.get():
            self.mg_line = None
            self.redraw()
            return
        if self.mg is None and (self.mg_thread is None
                                or not self.mg_thread.is_alive()):
            self.status.config(text="loading micrograd…")
            self.mg_line = None
            path = self.params_path

            def worker():
                try:
                    with open(path, "rb") as f:
                        f.seek(16 + 32)
                        flat = list(struct.unpack("<%dd" % (WTOT + BTOT),
                                                  f.read()))
                    self.mg = MGBridge(flat)
                    self.root.after(0, lambda: self.status.config(
                        text="micrograd loaded"))
                except Exception as e:
                    self.mg_var.set(False)
                    self.root.after(0, lambda: self.status.config(
                        text=f"micrograd error: {e}"))
            self.mg_thread = threading.Thread(target=worker, daemon=True)
            self.mg_thread.start()
        self.redraw()

    def mg_current(self):
        i, img = self.idx, self.imgs[self.idx]

        def worker():
            ym = self.mg.predict(img)
            pm = argmax(ym)
            if self.idx == i:
                self.mg_line = (pm, ym[pm])
                self.root.after(0, self.redraw)
        self.mg_thread = threading.Thread(target=worker, daemon=True)
        self.mg_thread.start()

    # ---- batch verify ----
    def verify_1000(self):
        self.btn_verify.config(state="disabled")
        self.status.config(text="verifying 1000…")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        n = min(1000, self.n)
        ok = 0
        for k in range(n):
            y = self.predict(self.imgs[k])
            am = argmax(y)
            self.batch_pred[k] = (am, y[am])
            ok += am == self.T[k]
        dt = (time.perf_counter() - t0) / n
        self.status.config(text=f"verified {n}: {ok} ok ({ok / n:.1%}), "
                                f"{dt * 1e6:.0f} us/image")
        self.btn_verify.config(state="normal")
        self.draw_strip()


def main():
    model = "build/model.bin"
    args = sys.argv[1:]
    if "--model" in args:
        model = args[args.index("--model") + 1]
    if "--selftest" in args:
        if not os.path.exists(model):
            print(f"model file not found: {model}")
            sys.exit(1)
        sys.exit(selftest(model))
    if not os.path.exists(model):
        print(f"model file not found: {model} "
              f"(training in progress? try --model build/model_pilot1.bin)")
        sys.exit(1)
    root = tk.Tk()
    root.geometry("1456x580+30+30")
    App(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
