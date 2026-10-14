/* test.c -- test driver for the pure-assembly MLP (src/neural.s).
 *
 * TDD: this driver was written BEFORE the assembly implementation.
 * Expected values come from test/ref.py (an independently verified Python
 * reference: its backprop is checked against finite differences, and its
 * math is cross-checked against a known-good project, see test_micrograd.py).
 *
 * Tests:
 *   1. forward/loss against frozen expected values (fixed weights)
 *   2. init: bit-exact mirror of the reference RNG (seed 77)
 *   3. determinism: same seed -> identical weights
 *   4. gradient check: nn_backward vs central finite differences (1e-6)
 *   5. XOR training: [2,8,8,1] 2-hidden-layer net converges to 4/4,
 *      loss trajectory matches the reference
 *
 * Build:  build.bat   (run from the project root)
 */
#include <stdio.h>
#include <string.h>
#include "expected.inc"

typedef struct net Net;

/* API implemented entirely in src/neural.s */
Net    *nn_create(int n_layers, const int sizes[], unsigned long long seed, const double *lr);
void    nn_free(Net *net);
void    nn_forward(const Net *net, const double *x, double *y);
double  nn_loss(const Net *net, const double *x, const double *t);
void    nn_backward(const Net *net, const double *x, const double *t, double *dW, double *dB);
void    nn_sgd_update(Net *net, const double *dW, const double *dB);
void    nn_get_weights(const Net *net, double *W, double *B);
void    nn_set_weights(Net *net, const double *W, const double *B);

static int g_pass = 0, g_fail = 0;

static double relerr(double a, double b)
{
    double d = a - b; if (d < 0) d = -d;
    double m = a > b ? a : b;
    if (m < 1e-300) m = 1e-300;
    return d / m;
}

#define CHECK(cond, ...) do {                                        \
    if (cond) { g_pass++; printf("    ok   " __VA_ARGS__); putchar('\n'); } \
    else      { g_fail++; printf("    FAIL " __VA_ARGS__); putchar('\n'); } \
} while (0)

#define SECTION(name) printf("== %s\n", name)

static int wtot(const int s[], int L)
{ int t = 0; for (int l = 0; l < L; l++) t += s[l] * s[l + 1]; return t; }

static int btot(const int s[], int L)
{ int t = 0; for (int l = 0; l < L; l++) t += s[l + 1]; return t; }

/* ------------------------------------------------------------------ */
/* 1. Forward / loss with fixed, binary-exact deterministic weights.  */
/*    sizes [2,3,2,1]: input(2) -> hidden(3) -> hidden(2) -> out(1)   */
/* ------------------------------------------------------------------ */
static const int S1[] = { 2, 3, 2, 1 };
#define L1 3
#define NW1 14   /* wtot(S1,L1) = 2*3 + 3*2 + 2*1 */
#define NB1 6    /* btot(S1,L1) = 3 + 2 + 1 */

/* same formula as ref.py section 1 (binary-exact: multiples of 2^-3, 2^-4) */
static void fill_fixed_weights(double *W, double *B)
{
    int k = 0, kb = 0;
    for (int l = 0; l < L1; l++) {
        int f = S1[l], o = S1[l + 1];
        for (int j = 0; j < o; j++)
            for (int i = 0; i < f; i++)
                W[k++] = 0.125 * ((2 * i + 3 * j + 5 * l + 1) % 7) - 0.4375;
        for (int j = 0; j < o; j++)
            B[kb++] = 0.0625 * (((j + 1) * (l + 1)) % 5) - 0.125;
    }
}

static void test_forward_fixed(void)
{
    SECTION("1. forward + loss (fixed weights, sizes [2,3,2,1])");
    double LR = 0.1;
    Net *n = nn_create(L1, S1, 1, &LR);
    if (!n) { CHECK(0, "nn_create returned NULL"); return; }

    double W[NW1], B[NB1];
    fill_fixed_weights(W, B);
    nn_set_weights(n, W, B);

    double x[2] = { 0.5, -0.75 }, t[1] = { 0.25 }, y[1];
    nn_forward(n, x, y);
    CHECK(relerr(y[0], EXP_FWD_Y) < 1e-12,
          "forward: y=%.17g expected %.17g (relerr %.3g)",
          y[0], EXP_FWD_Y, relerr(y[0], EXP_FWD_Y));

    double L = nn_loss(n, x, t);
    CHECK(relerr(L, EXP_FWD_LOSS) < 1e-12,
          "loss: %.17g expected %.17g (relerr %.3g)",
          L, EXP_FWD_LOSS, relerr(L, EXP_FWD_LOSS));

    nn_free(n);
}

/* ------------------------------------------------------------------ */
/* 2. Init mirror: xorshift64* + He uniform must be BIT-EXACT with    */
/*    the reference (no exp on this path, IEEE ops only).            */
/* ------------------------------------------------------------------ */
static void test_init_mirror(void)
{
    SECTION("2. init RNG mirror (seed 77, bit-exact)");
    double LR = 0.1;
    Net *n = nn_create(L1, S1, 77, &LR);
    if (!n) { CHECK(0, "nn_create returned NULL"); return; }

    double W[NW1], B[NB1];
    nn_get_weights(n, W, B);
    /* flat layout: W0[0..5], W1[6..11], W2[12..13]; B0[0..2], B1[3..4], B2[5] */
    int ok = W[0]  == EXP_INIT[0] &&   /* W[0][0] */
             W[4]  == EXP_INIT[1] &&   /* W[0][4] */
             W[9]  == EXP_INIT[2] &&   /* W[1][3] */
             W[13] == EXP_INIT[3];     /* W[2][1] */
    CHECK(ok, "W0_0=%.17g W0_4=%.17g W1_3=%.17g W2_1=%.17g",
          W[0], W[4], W[9], W[13]);
    CHECK(B[0] == 0.0 && B[3] == 0.0 && B[5] == 0.0, "biases initialized to 0");

    nn_free(n);
}

/* ------------------------------------------------------------------ */
/* 3. Determinism: two nets, same seed -> identical parameters.       */
/* ------------------------------------------------------------------ */
static void test_determinism(void)
{
    SECTION("3. determinism (same seed -> identical weights)");
    double LR = 0.1;
    Net *a = nn_create(L1, S1, 4242, &LR);
    Net *b = nn_create(L1, S1, 4242, &LR);
    if (!a || !b) { CHECK(0, "nn_create returned NULL");
        nn_free(a); nn_free(b); return; }

    double Wa[NW1], Wb[NW1];
    double Ba[NB1], Bb[NB1];
    nn_get_weights(a, Wa, Ba);
    nn_get_weights(b, Wb, Bb);
    CHECK(memcmp(Wa, Wb, sizeof Wa) == 0 && memcmp(Ba, Bb, sizeof Ba) == 0,
          "all %d W + %d B values identical", (int)(sizeof Wa / 8), (int)(sizeof Ba / 8));

    /* different seed must differ (sanity, not a strict contract) */
    Net *c = nn_create(L1, S1, 4243, &LR);
    double Wc[NW1], Bc[NB1];
    nn_get_weights(c, Wc, Bc);
    CHECK(memcmp(Wa, Wc, sizeof Wa) != 0, "different seed changes weights");

    nn_free(a); nn_free(b); nn_free(c);
}

/* ------------------------------------------------------------------ */
/* 4. Gradient check: analytic nn_backward vs central finite          */
/*    differences of nn_loss, eps = 1e-6. Max rel err must be < 1e-5. */
/* ------------------------------------------------------------------ */
static void test_gradient_check(void)
{
    SECTION("4. gradient check (backprop vs finite differences)");
    double LR = 0.1;
    Net *n = nn_create(L1, S1, 31337, &LR);
    if (!n) { CHECK(0, "nn_create returned NULL"); return; }

    /* guard: the net under test must match the frozen reference init */
    double W[NW1], B[NB1];
    nn_get_weights(n, W, B);
    int wok = 1, bok = 1;
    for (int i = 0; i < (int)sizeof W / 8; i++)
        if (W[i] != EXP_GC_W[i]) wok = 0;
    for (int i = 0; i < (int)sizeof B / 8; i++)
        if (B[i] != EXP_GC_B[i]) bok = 0;
    CHECK(wok && bok, "gradcheck net init matches oracle (guard)");
    if (!wok || !bok) { nn_free(n); return; }

    double x[2] = { 0.3, -0.6 }, t[1] = { 1.0 };
    double dW[NW1], dB[NB1];
    nn_backward(n, x, t, dW, dB);

    const double eps = 1e-6;
    double maxerr = 0.0;
    int nparam = 0;

    for (int k = 0; k < (int)sizeof W / 8; k++) {
        double p0 = W[k];
        double savedW[NW1], savedB[NB1];
        memcpy(savedW, W, sizeof W); memcpy(savedB, B, sizeof B);

        W[k] = p0 + eps; nn_set_weights(n, W, B); double lp = nn_loss(n, x, t);
        W[k] = p0 - eps; nn_set_weights(n, W, B); double lm = nn_loss(n, x, t);
        memcpy(W, savedW, sizeof W); memcpy(B, savedB, sizeof B);
        nn_set_weights(n, W, B);

        double num = (lp - lm) / (2.0 * eps);
        double ana = dW[k];
        double denom = (ana > 0 ? ana : -ana) + (num > 0 ? num : -num);
        if (denom < 1e-8) denom = 1e-8;
        double e = (ana - num < 0 ? ana - num : num - ana) / denom;
        if (e > maxerr) maxerr = e;
        nparam++;
    }
    for (int k = 0; k < (int)sizeof B / 8; k++) {
        double p0 = B[k];
        double savedW[NW1], savedB[NB1];
        memcpy(savedW, W, sizeof W); memcpy(savedB, B, sizeof B);

        B[k] = p0 + eps; nn_set_weights(n, W, B); double lp = nn_loss(n, x, t);
        B[k] = p0 - eps; nn_set_weights(n, W, B); double lm = nn_loss(n, x, t);
        memcpy(W, savedW, sizeof W); memcpy(B, savedB, sizeof B);
        nn_set_weights(n, W, B);

        double num = (lp - lm) / (2.0 * eps);
        double ana = dB[k];
        double denom = (ana > 0 ? ana : -ana) + (num > 0 ? num : -num);
        if (denom < 1e-8) denom = 1e-8;
        double e = (ana - num < 0 ? ana - num : num - ana) / denom;
        if (e > maxerr) maxerr = e;
        nparam++;
    }

    CHECK(maxerr < 1e-5, "max rel err = %.3e over %d params (limit 1e-5)",
          maxerr, nparam);

    /* SGD update must apply exactly W -= lr*grad for every parameter */
    int ok = 1;
    double W0[NW1], B0[NB1];
    memcpy(W0, W, sizeof W); memcpy(B0, B, sizeof B);
    nn_backward(n, x, t, dW, dB);
    nn_sgd_update(n, dW, dB);
    nn_get_weights(n, W, B);
    for (int i = 0; i < (int)sizeof W / 8; i++) {
        if (W[i] != W0[i] - 0.1 * dW[i]) ok = 0;
    }
    for (int i = 0; i < (int)sizeof B / 8; i++) {
        if (B[i] != B0[i] - 0.1 * dB[i]) ok = 0;
    }
    CHECK(ok, "sgd update: W -= lr*grad applied exactly");

    nn_free(n);
}

/* ------------------------------------------------------------------ */
/* 5. XOR: [2,8,8,1] (two hidden layers), seed 20261014, lr 0.05.     */
/*    Full-batch SGD for 3000 epochs. Must reach 4/4 and the loss     */
/*    trajectory must match the reference within 1e-6 relative.       */
/* ------------------------------------------------------------------ */
static const int SX[] = { 2, 8, 8, 1 };
#define LX 3
#define EPOCHS 3000
#define NWX 88   /* wtot(SX,LX) = 2*8 + 8*8 + 8*1 */
#define NBX 17   /* btot(SX,LX) = 8 + 8 + 1 */

static double xor_batch_loss(Net *n)
{
    static const double X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    static const double T[4]    = {0.0, 1.0, 1.0, 0.0};
    double s = 0.0;
    for (int i = 0; i < 4; i++) s = s + nn_loss(n, X[i], &T[i]);
    return s;
}

static void xor_batch_grad(Net *n, double *GW, double *GB)
{
    static const double X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    static const double T[4]    = {0.0, 1.0, 1.0, 0.0};
    memset(GW, 0, wtot(SX, LX) * 8);
    memset(GB, 0, btot(SX, LX) * 8);
    double gW[NWX], gB[NBX];
    for (int i = 0; i < 4; i++) {
        nn_backward(n, X[i], &T[i], gW, gB);
        for (int k = 0; k < wtot(SX, LX); k++) GW[k] += gW[k];
        for (int k = 0; k < btot(SX, LX); k++) GB[k] += gB[k];
    }
}

static void test_xor(void)
{
    SECTION("5. XOR training ([2,8,8,1], 3000 epochs, full-batch SGD)");
    double LR = 0.05;
    Net *n = nn_create(LX, SX, 20261014, &LR);
    if (!n) { CHECK(0, "nn_create returned NULL"); return; }

    double GW[NWX], GB[NBX];

    double loss0 = xor_batch_loss(n);
    CHECK(relerr(loss0, EXP_XOR_LOSS0) < 1e-12,
          "loss before training = %.17g (expected %.17g)", loss0, EXP_XOR_LOSS0);

    double loss100 = 0, loss1000 = 0, lossf = 0;
    for (int e = 1; e <= EPOCHS; e++) {
        xor_batch_grad(n, GW, GB);
        nn_sgd_update(n, GW, GB);
        if (e == 100)  loss100  = xor_batch_loss(n);
        if (e == 1000) loss1000 = xor_batch_loss(n);
    }
    lossf = xor_batch_loss(n);

    CHECK(relerr(loss100, EXP_XOR_LOSS100) < 1e-6,
          "loss@100  = %.17g (expected %.17g, relerr %.3g)",
          loss100, EXP_XOR_LOSS100, relerr(loss100, EXP_XOR_LOSS100));
    CHECK(relerr(loss1000, EXP_XOR_LOSS1000) < 1e-6,
          "loss@1000 = %.17g (expected %.17g, relerr %.3g)",
          loss1000, EXP_XOR_LOSS1000, relerr(loss1000, EXP_XOR_LOSS1000));
    CHECK(relerr(lossf, EXP_XOR_LOSS_FINAL) < 1e-6,
          "loss@3000 = %.17g (expected %.17g, relerr %.3g)",
          lossf, EXP_XOR_LOSS_FINAL, relerr(lossf, EXP_XOR_LOSS_FINAL));
    CHECK(loss0 > loss100 && loss100 > loss1000 && loss1000 > lossf,
          "loss strictly decreases: %.4g > %.4g > %.4g > %.4g",
          loss0, loss100, loss1000, lossf);
    CHECK(lossf < 0.01, "final loss < 0.01 (got %.6g)", lossf);

    static const double X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    static const double T[4]    = {0.0, 1.0, 1.0, 0.0};
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        double y[1];
        nn_forward(n, X[i], y);
        double pred = y[0] > 0.5 ? 1.0 : 0.0;
        if (pred == T[i]) correct++;
    }
    CHECK(correct == 4, "XOR accuracy = %d/4", correct);

    nn_free(n);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    test_forward_fixed();
    test_init_mirror();
    test_determinism();
    test_gradient_check();
    test_xor();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
