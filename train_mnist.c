/* train_mnist.c -- train the pure-assembly MLP on MNIST: 784 -> 256 -> 256 -> 10.
 *
 * Mini-batch plain SGD on the asm math contract (see src/neural.s header):
 * each batch of `batch` samples calls nn_backward per sample (it writes
 * FRESH per-sample gradients), accumulates them in flat C buffers, then
 * applies one update.  The net's lr is set to lr/batch, so
 *   W -= (lr/batch) * sum_s grad_s  ==  W -= lr * mean_s grad_s.
 * (batch == n_train reproduces plain full-batch SGD.)
 *
 * A checkpoint is written after every epoch: <model_out>_e<epoch>.bin,
 * so the best epoch can be selected afterwards.
 *
 * Usage:
 *   train_mnist.exe <train_imgs.bin> <train_lbls.bin> <n_train> \
 *                   <epochs> <lr> <batch> <seed> <model_out.bin> \
 *                   <test_imgs.bin> <test_lbls.bin> <n_test>
 *
 * Prints one line per epoch (seconds for train+updates, test accuracy,
 * test loss) and saves the trained weights to model_out.bin:
 *   int32 L, int32 sizes[8], int32 wtot, int32 btot,
 *   wtot doubles W (row-major per layer), btot doubles B.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

/* API implemented entirely in src/neural.s (win64, int-class args). */
extern void  *nn_create(int L, const int *sizes, unsigned long long seed, const double *lr);
extern void   nn_free(void *n);
extern void   nn_forward(const void *n, const double *x, double *y);
extern double nn_loss(const void *n, const double *x, const double *t);
extern void   nn_backward(const void *n, const double *x, const double *t, double *dW, double *dB);
extern void   nn_sgd_update(void *n, const double *dW, const double *dB);
extern void   nn_get_weights(const void *n, double *W, double *B);

#define L 3
static const int SIZES[L + 1] = {784, 256, 256, 10};

static int g_wtot, g_btot;

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(2); }
    return p;
}

static void *xread(const char *path, long n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    void *p = xmalloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read on %s\n", path); exit(2);
    }
    fclose(f);
    return p;
}

/* raw uint8 images -> doubles 0..1 (/255; the asm He init assumes O(1) inputs) */
static double *load_imgs(const char *path, int n)
{
    unsigned char *raw = xread(path, (long)n * 784);
    double *x = xmalloc((size_t)n * 784 * sizeof(double));
    for (int i = 0; i < n * 784; i++) x[i] = (double)raw[i] / 255.0;
    free(raw);
    return x;
}

static double onehot[10][10] = {0};

static void eval_net(const void *n, const double *X, const unsigned char *T,
                     int cnt, int *correct, double *loss)
{
    double y[10];
    int c = 0;
    double sloss = 0.0;
    for (int s = 0; s < cnt; s++) {
        nn_forward(n, X + (size_t)s * 784, y);
        int pred = 0;
        for (int k = 1; k < 10; k++) if (y[k] > y[pred]) pred = k;
        if (pred == T[s]) c++;
        double d = 0.0;
        for (int k = 0; k < 10; k++) {
            double e = y[k] - onehot[T[s]][k];
            d += e * e;
        }
        sloss += 0.5 * d;
    }
    *correct = c;
    *loss = sloss / cnt;
}

static void save_model(const void *n, const int *sizes, const char *path)
{
    double *W = xmalloc((size_t)g_wtot * sizeof(double));
    double *B = xmalloc((size_t)g_btot * sizeof(double));
    nn_get_weights(n, W, B);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    int hdr[4] = {L, 0, g_wtot, g_btot};
    fwrite(hdr, sizeof(int), 4, f);
    fwrite(sizes, sizeof(int), 8, f);
    fwrite(W, sizeof(double), (size_t)g_wtot, f);
    fwrite(B, sizeof(double), (size_t)g_btot, f);
    fclose(f);
    free(W); free(B);
}

int main(int argc, char **argv)
{
    if (argc != 12) {
        fprintf(stderr,
                "usage: train_mnist <tr_imgs> <tr_lbls> <n_train> <epochs> "
                "<lr> <batch> <seed> <model_out> <te_imgs> <te_lbls> <n_test>\n");
        return 2;
    }
    int n_train  = atoi(argv[3]);
    int epochs   = atoi(argv[4]);
    double lr    = atof(argv[5]);
    int batch    = atoi(argv[6]);
    unsigned long long seed = strtoull(argv[7], 0, 10);
    int n_test = atoi(argv[11]);
    if (batch < 1 || batch > n_train) { fprintf(stderr, "bad batch\n"); return 2; }

    printf("mnist asm training: %d train samples, %d epochs, lr=%.4f, batch=%d, seed=%llu\n",
           n_train, epochs, lr, batch, seed);
    fflush(stdout);

    double *X  = load_imgs(argv[1], n_train);
    unsigned char *T = xread(argv[2], n_train);
    double *XE  = load_imgs(argv[9], n_test);
    unsigned char *TE = xread(argv[10], n_test);

    for (int a = 0; a < 10; a++) onehot[a][a] = 1.0;
    for (int l = 0; l < L; l++) { g_wtot += SIZES[l] * SIZES[l + 1]; g_btot += SIZES[l + 1]; }

    LARGE_INTEGER fq; QueryPerformanceFrequency(&fq);

    int sizes[8] = {0};
    for (int i = 0; i <= L; i++) sizes[i] = SIZES[i];
    double lr_net = lr / (double)batch;
    void *n = nn_create(L, sizes, seed, &lr_net);
    if (!n) { fprintf(stderr, "nn_create failed\n"); return 2; }

    double *GW = xmalloc((size_t)g_wtot * sizeof(double));
    double *GB = xmalloc((size_t)g_btot * sizeof(double));
    double *gW = xmalloc((size_t)g_wtot * sizeof(double));
    double *gB = xmalloc((size_t)g_btot * sizeof(double));

    for (int e = 1; e <= epochs; e++) {
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        for (int b0 = 0; b0 < n_train; b0 += batch) {
            int b1 = b0 + batch;
            if (b1 > n_train) b1 = n_train;
            /* first sample seeds the accumulator (avoids a memset) */
            nn_backward(n, X + (size_t)b0 * 784, onehot[T[b0]], gW, gB);
            memcpy(GW, gW, (size_t)g_wtot * sizeof(double));
            memcpy(GB, gB, (size_t)g_btot * sizeof(double));
            for (int s = b0 + 1; s < b1; s++) {
                nn_backward(n, X + (size_t)s * 784, onehot[T[s]], gW, gB);
                for (int k = 0; k < g_wtot; k++) GW[k] += gW[k];
                for (int k = 0; k < g_btot; k++) GB[k] += gB[k];
            }
            nn_sgd_update(n, GW, GB);
        }
        QueryPerformanceCounter(&t1);
        double dt_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)fq.QuadPart;

        int correct; double loss;
        eval_net(n, XE, TE, n_test, &correct, &loss);
        printf("epoch %3d  %8.1f ms  acc %5d/%d (%.2f%%)  test_loss %.6f\n",
               e, dt_ms, correct, n_test, 100.0 * correct / n_test, loss);
        fflush(stdout);

        char ckpt[4096];
        snprintf(ckpt, sizeof ckpt, "%s_e%d.bin", argv[8], e);
        save_model(n, sizes, ckpt);
    }

    /* final model */
    save_model(n, sizes, argv[8]);
    printf("model written to %s (%d + %d params)\n", argv[8], g_wtot, g_btot);
    fflush(stdout);

    free(GW); free(GB); free(gW); free(gB);
    free(X); free(T); free(XE); free(TE);
    nn_free(n);
    return 0;
}
