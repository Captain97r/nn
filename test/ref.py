#!/usr/bin/env python3
"""
Reference oracle for the x86-64 assembly MLP (src/neural.s).

Mirrors the assembly algorithm step-for-step (same formulas, same
summation order, IEEE double precision):

  * RNG      : xorshift64* (Vigna), seeded with one splitmix64 mix step
  * Init     : He uniform, W[l][j*f+i] = lim * (2*u - 1), lim = sqrt(6/(f+o))
               B = 0
  * Forward  : z_j = (sum_i W[j*f+i]*a_i) + b_j
               hidden layers: ReLU(z);  output layer: sigmoid(z)
  * Loss     : L = 0.5 * sum_j (y_j - t_j)^2
  * Backprop : dL/dW[l][j*f+i] = delta_j * a_i ; dL/dB[l][j] = delta_j
               delta_out[j]   = (y_j - t_j) * y_j * (1 - y_j)
               delta_in[i]    = (sum_j W[j*f+i] * delta_j) * (a_i > 0)
  * SGD      : full batch, W -= lr * G, B -= lr * G

Run:
  python ref.py            -> verifies the oracle itself (finite-difference
                              gradient check), trains the XOR reference net,
                              writes test/expected.inc with expected values
  python ref.py --check    -> gradient check only, no .inc written
"""
import math
import sys

MASK = (1 << 64) - 1
SM_CONST = 0x9E3779B97F4A7C15


def splitmix64_mix(x: int) -> int:
    x = (x + SM_CONST) & MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


class Rng:
    """xorshift64* with splitmix64 seeding (identical to the asm version)."""

    __slots__ = ("s",)

    def __init__(self, seed: int):
        self.s = splitmix64_mix(seed & MASK) or 1

    def u64(self) -> int:
        s = self.s
        s ^= s >> 12
        s ^= (s << 25) & MASK
        s ^= s >> 27
        self.s = s & MASK
        return (s * 0x2545F4914F6CDD1D) & MASK

    def unif(self) -> float:
        """Uniform in [0, 1): top 53 bits scaled by 2^-53."""
        return (self.u64() >> 11) * 2.0 ** -53


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class MLP:
    """L fully-connected layers; ReLU hidden, sigmoid output."""

    def __init__(self, sizes, seed, lr):
        self.L = len(sizes) - 1
        self.sizes = list(sizes)
        rng = Rng(seed)
        self.W, self.B = [], []
        for l in range(self.L):
            f, o = self.sizes[l], self.sizes[l + 1]
            lim = math.sqrt(6.0 / (f + o))
            row = []
            for j in range(o):
                for i in range(f):
                    row.append(lim * (2.0 * rng.unif() - 1.0))
            self.W.append(row)
            self.B.append([0.0] * o)
        self.lr = lr

    # ---- forward, keeping per-layer activations ---------------------------
    def forward(self, x):
        a = list(x)
        acts = [a]
        for l in range(self.L):
            f, o = self.sizes[l], self.sizes[l + 1]
            Wl, Bl = self.W[l], self.B[l]
            z = []
            for j in range(o):
                s = 0.0
                for i in range(f):
                    s = s + Wl[j * f + i] * a[i]
                s = s + Bl[j]
                z.append(s)
            if l == self.L - 1:
                a = [sigmoid(v) for v in z]
            else:
                a = [v if v > 0.0 else 0.0 for v in z]
            acts.append(a)
        return a, acts

    def loss(self, x, t):
        y, _ = self.forward(x)
        s = 0.0
        for j in range(len(y)):
            d = y[j] - t[j]
            s = s + d * d
        return 0.5 * s

    # ---- backprop for one sample, gradients returned as flat lists --------
    def backward(self, x, t):
        y, acts = self.forward(x)
        dW = [[0.0] * (o * f) for f, o in zip(self.sizes[:-1], self.sizes[1:])]
        dB = [[0.0] * o for o in self.sizes[1:]]
        # output delta
        delta = [(y[j] - t[j]) * y[j] * (1.0 - y[j]) for j in range(self.sizes[-1])]
        for l in range(self.L - 1, -1, -1):
            f, o = self.sizes[l], self.sizes[l + 1]
            Wl, Bl, a_in = self.W[l], self.B[l], acts[l]
            for j in range(o):
                d = delta[j]
                if d != 0.0:
                    for i in range(f):
                        dW[l][j * f + i] = dW[l][j * f + i] + d * a_in[i]
                dB[l][j] = dB[l][j] + d
            if l > 0:
                a_prev = acts[l]
                new = [0.0] * f
                for i in range(f):
                    if a_prev[i] > 0.0:
                        s = 0.0
                        for j in range(o):
                            s = s + Wl[j * f + i] * delta[j]
                        new[i] = s
                delta = new
        return dW, dB

    def sgd_update(self, dW, dB):
        for l in range(self.L):
            f, o = self.sizes[l], self.sizes[l + 1]
            for j in range(o):
                for i in range(f):
                    self.W[l][j * f + i] = self.W[l][j * f + i] - self.lr * dW[l][j * f + i]
                self.B[l][j] = self.B[l][j] - self.lr * dB[l][j]


def gradcheck(m: MLP, x, t, eps=1e-6):
    """Analytic (backprop) vs central-difference gradients. Returns max rel err."""
    dW_a, dB_a = m.backward(x, t)
    L = 0
    maxerr = 0.0
    for l in range(m.L):
        for k in range(len(m.W[l])):
            p0 = m.W[l][k]
            m.W[l][k] = p0 + eps
            lp = m.loss(x, t)
            m.W[l][k] = p0 - eps
            lm = m.loss(x, t)
            m.W[l][k] = p0
            num = (lp - lm) / (2.0 * eps)
            ana = dW_a[l][k]
            err = abs(ana - num) / max(1e-8, abs(ana) + abs(num))
            maxerr = max(maxerr, err)
            L += 1
        for j in range(len(m.B[l])):
            p0 = m.B[l][j]
            m.B[l][j] = p0 + eps
            lp = m.loss(x, t)
            m.B[l][j] = p0 - eps
            lm = m.loss(x, t)
            m.B[l][j] = p0
            num = (lp - lm) / (2.0 * eps)
            ana = dB_a[l][j]
            err = abs(ana - num) / max(1e-8, abs(ana) + abs(num))
            maxerr = max(maxerr, err)
            L += 1
    return maxerr, L


def fmt(v):
    return f"{v!r}"


def main():
    check_only = "--check" in sys.argv

    # ------------------------------------------------------------------
    # 0) Self-verify the oracle's backprop with finite differences
    # ------------------------------------------------------------------
    gm = MLP([2, 3, 2, 1], seed=31337, lr=0.1)
    gx, gt = [0.3, -0.6], [1.0]
    err, nparam = gradcheck(gm, gx, gt)
    print(f"oracle gradcheck: max rel err = {err:.3e} over {nparam} params")
    if err > 1e-6:
        print("ORACLE BACKPROP IS WRONG")
        sys.exit(1)
    if check_only:
        return

    # ------------------------------------------------------------------
    # 1) Fixed-weight forward / loss test vectors
    #    sizes [2,3,2,1]; weights deterministic & binary-exact
    # ------------------------------------------------------------------
    fx = [2, 3, 2, 1]
    fW = []
    fB = []
    for l in range(len(fx) - 1):
        f, o = fx[l], fx[l + 1]
        fW.append([0.125 * ((2 * i + 3 * j + 5 * l + 1) % 7) - 0.4375
                   for j in range(o) for i in range(f)])
        fB.append([0.0625 * (((j + 1) * (l + 1)) % 5) - 0.125 for j in range(o)])
    fm = MLP(fx, seed=0, lr=0.1)
    fm.W, fm.B = fW, fB
    fxin = [0.5, -0.75]
    ftgt = [0.25]
    fy, _ = fm.forward(fxin)
    floss = fm.loss(fxin, ftgt)
    print(f"fixed-weight forward: y = {fy[0]!r}, loss = {floss!r}")

    # ------------------------------------------------------------------
    # 2) RNG / init mirror: exact expected values (no exp on this path)
    # ------------------------------------------------------------------
    im = MLP([2, 3, 2, 1], seed=77, lr=0.1)
    init_vals = (im.W[0][0], im.W[0][4], im.W[1][3], im.W[2][1])
    print(f"init mirror (seed 77): {[f'{v!r}' for v in init_vals]}")

    # ------------------------------------------------------------------
    # 3) XOR training reference: sizes [2,8,8,1], seed 20261014, lr 0.05
    # ------------------------------------------------------------------
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    T = [0.0, 1.0, 1.0, 0.0]
    xm = MLP([2, 8, 8, 1], seed=20261014, lr=0.05)

    def batch_grad():
        GW = [[0.0] * (o * f) for f, o in zip(xm.sizes[:-1], xm.sizes[1:])]
        GB = [[0.0] * o for o in xm.sizes[1:]]
        for xi, ti in zip(X, T):
            dW, dB = xm.backward(xi, [ti])
            for l in range(xm.L):
                for k in range(len(GW[l])):
                    GW[l][k] += dW[l][k]
                for j in range(len(GB[l])):
                    GB[l][j] += dB[l][j]
        return GW, GB

    def batch_loss():
        s = 0.0
        for xi, ti in zip(X, T):
            s = s + xm.loss(xi, [ti])
        return s

    loss0 = batch_loss()
    EPOCHS = 3000
    rec = {100: None, 1000: None}
    for e in range(1, EPOCHS + 1):
        GW, GB = batch_grad()
        xm.sgd_update(GW, GB)
        if e in rec:
            rec[e] = batch_loss()
    loss_final = batch_loss()
    outs = [xm.forward(xi)[0][0] for xi in X]
    acc = sum(1 for o, t in zip(outs, T) if (1.0 if o > 0.5 else 0.0) == t) / len(T)
    print(f"XOR reference: loss0={loss0!r} loss100={rec[100]!r} "
          f"loss1000={rec[1000]!r} loss_final={loss_final!r} acc={acc}")
    if acc != 1.0 or loss_final > 0.01:
        print("REFERENCE XOR NET FAILED TO CONVERGE -- tune before freezing")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4) Gradient-check reference values are NOT needed (the C driver does
    #    its own finite differences against nn_backward), but we freeze the
    #    net's initial weights so the driver can compare its backprop on a
    #    net with known contents.
    # ------------------------------------------------------------------
    gcW = [v for row in gm.W for v in row]
    gcB = [v for row in gm.B for v in row]

    # ------------------------------------------------------------------
    # Write expected.inc
    # ------------------------------------------------------------------
    with open("expected.inc", "w") as f:
        w = f.write
        w("// Auto-generated by test/ref.py -- do not edit.\n")
        w(f"static const double EXP_FWD_Y = {fmt(fy[0])};\n")
        w(f"static const double EXP_FWD_LOSS = {fmt(floss)};\n")
        w("static const double EXP_INIT[4] = {\n")
        for v in init_vals:
            w(f"    {fmt(v)},\n")
        w("};\n")
        w(f"static const double EXP_XOR_LOSS0 = {fmt(loss0)};\n")
        w(f"static const double EXP_XOR_LOSS100 = {fmt(rec[100])};\n")
        w(f"static const double EXP_XOR_LOSS1000 = {fmt(rec[1000])};\n")
        w(f"static const double EXP_XOR_LOSS_FINAL = {fmt(loss_final)};\n")
        w(f"static const double EXP_GC_W[{len(gcW)}] = {{\n")
        for v in gcW:
            w(f"    {fmt(v)},\n")
        w("};\n")
        w(f"static const double EXP_GC_B[{len(gcB)}] = {{\n")
        for v in gcB:
            w(f"    {fmt(v)},\n")
        w("};\n")
    print("wrote expected.inc")


if __name__ == "__main__":
    main()
