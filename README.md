# nn — two-layer perceptron in pure x86-64 assembly (Windows)

A fully-connected MLP with **two hidden layers** (ReLU), a sigmoid output,
MSE loss and full-batch SGD training, implemented in **pure x86-64 AT&T
assembly** (`src/neural.s`) for Windows. No C in the network itself — the
only imports from the CRT are `malloc`, `free` and `exp`. The test driver
(`test/test.c`) is plain C and is test glue, not part of the network.

Developed **test-first**: the test suite and an independent Python oracle
were written and frozen *before* the assembly, so the expected values in
`test/expected.inc` could not be "adjusted" to fit a buggy implementation.

## Build

Requires a MinGW-w64 toolchain (built with w64devkit at `D:\w64devkit`;
override with `set W64=<dir>`).

```
build.bat
```

produces `build/neural.o` (assembled with `x86_64-w64-mingw32-as`) and
`build/test.exe` (test driver linked against the object).

## Run

```
build\test.exe                 :: 16-check asm test suite
python test\test_micrograd.py  :: cross-verification vs micrograd (needs micrograd/ clone)
```

## Network

Sizes are given as `sizes[0..L]` (input .. output). The tests use:

- fixed-weight net `[2,3,2,1]` (2 hidden layers) for forward/loss/gradient checks
- XOR net `[2,8,8,1]`, seed 20261014, lr 0.05, full-batch, 3000 epochs

Math (per layer `l`, output neuron `j`):

```
z_l[j] = sum_i W_l[j][i] * a_{l-1}[i] + B_l[j]
a_l[j] = ReLU(z_l[j])                        (hidden layers)
y[j]   = sigmoid(z_L[j]) = 1/(1+exp(-z_L[j])) (output)
L      = 0.5 * sum_j (y[j] - t[j])^2
```

Backprop (MSE + sigmoid output): `delta_out = (y - t) * y * (1 - y)`,
hidden deltas gate on `a_prev > 0`. SGD: `W -= lr * grad` in the same
layer/parameter order the gradients were computed in.

Init: He-style uniform `W ~ +/- sqrt(6/(in+out))` from a xorshift64* RNG
(state mixed with a splitmix64 step so seed 0 works); biases zero.
Uniform draw: `(state >> 11) * 2^-53`.

## API (win64, C linkage)

| symbol | signature |
|---|---|
| `nn_create` | `(int L, const int *sizes, uint64 seed, const double *lr) -> void*` |
| `nn_free`   | `(void *n)` |
| `nn_forward`| `(const void *n, const double *x, double *y)` |
| `nn_loss`   | `(const void *n, const double *x, const double *t) -> double` (0.5*sum sq) |
| `nn_backward`| `(const void *n, const double *x, const double *t, double *dW, double *dB)` — flat, all W then all B, row-major; buffers zeroed first |
| `nn_sgd_update` | `(void *n, const double *dW, double *dB)` |
| `nn_get_weights` / `nn_set_weights` | `(void *n, double *W, double *B)` / `(void *n, const double *W, const double *B)` — flat copies, test hooks |

Limits: `L <= 7`, any layer size `<= 256` (static delta buffers),
32-bit index arithmetic for the flat buffers.

## Test suite (test/test.c — written before the assembly)

1. **T1 fixed-weight forward + loss** — `fill_fixed_weights` for `[2,3,2,1]`,
   input `{0.5,-0.75}`, target `{0.25}`; output and loss must match the frozen
   oracle values to relerr < 1e-12. (Actual: relerr 0 — bit-exact.)
2. **T2 RNG mirror** — 4 spot-checked weights of a `[2,3,2,1]` net, seed 77,
   must equal the Python xorshift64* mirror bit-for-bit.
3. **T3 determinism** — same seed ⇒ identical weights; different seed ⇒ different.
4. **T4 gradient check** — analytic backprop vs central finite differences
   (eps 1e-6, 20 params of a seed-31337 `[2,3,2,1]` net) to relerr < 1e-5,
   plus an exact SGD update check. (Actual: max relerr < 5e-6.)
5. **T5 XOR training** — loss checkpoints (0/100/1000/3000) match the frozen
   oracle to relerr < 1e-6, loss strictly decreases, final loss < 0.01,
   accuracy 4/4. (Actual: @100 bit-exact, @1000/@3000 within ~1 ulp.)

`test/ref.py` is the independent oracle (same formulas, IEEE-754 doubles,
identical operation order); it generated `test/expected.inc`.

**Result: 16 passed, 0 failed.** Trajectory fidelity, measured by dumping
all 12004 per-sample losses of the full 3000-epoch XOR run from the built
binary and bit-comparing against the oracle: **7878 values bit-exact; the
4126 that differ are all within 255 ulp (≤ ~4e-14 relative)**. The
difference source is the CRT `exp`: MinGW's libm and CPython's `math.exp`
disagree by exactly 1 ulp on a small fraction of arguments (7 of 1601 in a
direct side-by-side probe), and that 1-ulp wobble feeds back through SGD
(y → delta → weights), so the two trajectories walk ~1e-14 apart from
around epoch 30 onward. Every other operation (accumulation order, ReLU,
sigmoid arithmetic, backprop, SGD, init) is bit-identical — the test
tolerance windows (1e-6) encode this, and it is a platform libm difference,
not an arithmetic bug.

## Cross-verification vs a well-known project

`micrograd/` is a clone of [karpathy/micrograd](https://github.com/karpathy/micrograd)
(widely known, dependency-free autograd engine). `test/test_micrograd.py`
builds the **same architecture** (ReLU hidden layers, sigmoid output via a
local `exp` extension written in micrograd's own `_backward` pattern — the
cloned repo is unmodified) from micrograd `Value`s, reuses the same fixed
weights / same seed-31337 init / same frozen constants, and runs the same
test logic with the same criteria:

1. **forward + loss** on the T1 fixed weights vs the frozen expected values
   (independently validates the constants the asm was tested against) —
   relerr 0.
2. **gradient check**: micrograd's autograd `.grad` vs central finite
   differences, max relerr < 1e-5 — got 6.99e-11.
3. **XOR training**: same init/loss/SGD criteria — checkpoints match the
   frozen oracle to ~1e-16, strictly decreasing, final loss < 0.01, 4/4.

**Result: 11 passed, 0 failed** — both the assembly implementation and the
reference project pass the identical test suite.

## Layout

```
src/neural.s              pure-assembly MLP (the implementation)
test/test.c               C test driver (T1-T5), links neural.o
test/expected.inc         frozen expected values (generated by ref.py)
test/ref.py               independent Python oracle
test/test_micrograd.py    cross-verification driver vs micrograd
micrograd/                karpathy/micrograd clone (unmodified)
build.bat                 build script (as + gcc, w64devkit)
```
