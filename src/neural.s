# neural.s -- two-hidden-layer fully-connected MLP, pure x86-64 assembly
# (GNU as / GAS, win64 ABI, built by D:\w64devkit's as.exe).
#
# Math contract (MUST stay bit-identical to the Python oracle in test/ref.py,
# which mirrors this arithmetic double-by-double in the same order):
#
#   init     W[o][i] = lim * (2*u01 - 1.0),  u01 = (u64() >> 11) * 2^-53,
#            lim = sqrt(6.0/(in+out));       B = 0.0
#   rng      xorshift64*:  s ^= s>>12; s ^= s<<25; s ^= s>>27;  u = (s *
#            0x2545F4914F6CDD1D) & MASK64;  seed pre-mixed with splitmix64
#            finalizer (or 1 if the result is 0)
#   forward  s = 0.0;  s += W[i]*a_in[i] (i ascending);  s += B[j]
#            hidden: ReLU   v > +0.0 ? v : +0.0
#            output: 1/(1+exp(-v))   (exp imported from the default CRT)
#   loss     0.5 * sum((y-t)^2)
#   backward delta_out[j] = (y-t) * y * (1-y)
#            dB[j] += d (always)
#            if d != +0.0/-0.0:  dW[j*in+i] += d * a_in[i]
#            delta_in[i] = a_prev[i] > +0.0 ? sum_j W[j*in+i]*delta[j]
#                          (j ascending) : +0.0
#   sgd      W = W - lr*grad   (scalar: t = lr*g; W -= t)
#
# Layout: one malloc of (192 + total*8) bytes.  The net struct occupies
# the first 192 bytes (offsets below; the struct is accessed with
# unaligned-safe instructions, so only lr must be a real double); the
# data region (W blocks, then B blocks, then A blocks; total = wtot+btot+
# atot doubles) follows at +0xC0.  wtot = sum sizes[l]*sizes[l+1],
# btot = sum sizes[l+1], atot = sum sizes[0..L]:
#
#   offset  member
#   0x00    double lr
#   0x08    int   n      (L)
#   0x10    uint64 rng
#   0x18    void*  buf   (data region base = net + 0xC0)
#   0x20    int   sizes[8]
#   0x40    void*  W[8]    (W[l] row-major, sizes[l]*sizes[l+1] doubles)
#   0x60    void*  A[8]    (activations; A[0]=input ptr, A[L]=output)
#   0x80    void*  B[8]
#   0xA0    int   size     (total doubles in buf)
#   0xA4    pad          -> sizeof = 192
#
# API (C linkage, win64 calling convention; first 4 args rcx/rdx/r8/r9 --
# all int-class, so slot assignment is identical under the standard MS ABI
# and w64devkit gcc 16.2's unified-slot ABI; 32-byte shadow space honored
# before every call):
#   void*  nn_create(int L, const int *sizes, uint64 seed, const double *lr)
#   void   nn_free(void *n)
#   void   nn_forward(const void *n, const double *x, double *y)
#   double nn_loss(const void *n, const double *x, const double *t)
#   void   nn_backward(const void *n, const double *x, const double *t,
#                      double *dW, double *dB)   ; flat buffers, zeroed
#   void   nn_sgd_update(void *n, const double *dW, const double *dB)
#   void   nn_get_weights(const void *n, double *W, double *B)
#   void   nn_set_weights(void *n, const double *W, const double *B)
#
# Limits: L <= 7 (8 slot arrays); any single layer size <= 256 units
# (static delta buffers); total parameter count must keep 32-bit index math
# (practically unbounded for tests).

    .file   "neural.s"
    .text

    .set NET_N,     0x08
    .set NET_LR,    0x00
    .set NET_RNG,   0x10
    .set NET_BUF,   0x18
    .set NET_SIZES, 0x20
    .set NET_W,     0x40
    .set NET_A,     0x60
    .set NET_B,     0x80
    .set NET_SIZE,  0xA0

    .globl nn_create
    .globl nn_free
    .globl nn_forward
    .globl nn_loss
    .globl nn_backward
    .globl nn_sgd_update
    .globl nn_get_weights
    .globl nn_set_weights

    .extern malloc
    .extern free
    .extern exp

    .section .rdata
    .align 16
.LC_0:    .quad 0x0000000000000000          # +0.0
.LC_1:    .quad 0x3FF0000000000000          # 1.0
.LC_2:    .quad 0x4000000000000000          # 2.0
.LC_05:   .quad 0x3FE0000000000000          # 0.5
.LC_6:    .quad 0x4018000000000000          # 6.0
.LC_2m53: .quad 0x3CA0000000000000          # 2^-53
    .align 16
.LK_SM:   .quad 0x9E3779B97F4A7C15          # splitmix64 increment
.LK_K1:   .quad 0xBF58476D1CE4E5B9
.LK_K2:   .quad 0x94D049BB133111EB
.LK_GLD:  .quad 0x2545F4914F6CDD1D          # xorshift64* multiplier

    .text
# ---------------------------------------------------------------------------
# shared forward pass: nn in rcx, x in rdx.  Writes activations into the
# net's A slots (A[0] = x), leaves y = A[L] in rax.  Clobbers: rcx, rdx,
# rbx, r8-r15, xmm0-2 (caller-saved).
# ---------------------------------------------------------------------------
    .p2align 4
.Ldo_forward:
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    pushq %rsi
    subq $88, %rsp                    # 32 shadow (exp) + 56 scratch
    movq %rcx, %r14                   # net
    movq %rdx, %r12                   # x
    movq %r12, NET_A(%r14)            # A[0] = x
    leaq NET_A(%r14), %r13            # A slot base
    movl NET_N(%r14), %r10d
    decl %r10d                        # last = L-1
    xorl %ebx, %ebx                   # l = 0
.Lfl:
    movl NET_SIZES(%r14,%rbx,4), %ecx     # f
    movq NET_W(%r14,%rbx,8), %r11         # Wl
    movq NET_B(%r14,%rbx,8), %r15         # Bl
    movq NET_A(%r14,%rbx,8), %r12         # a_in
    movq 8(%r13,%rbx,8), %r8           # a_out = A[l+1] (data ptr)
    movl %ebx, %esi
    incl %esi
    movl NET_SIZES(%r14,%rsi,4), %edx     # o
    xorl %eax, %eax                       # j = 0
.Lfj:
    movl %eax, %esi
    imull %ecx, %esi                  # j*f
    leaq 0(%r11,%rsi,8), %r9          # row cursor
    movsd .LC_0(%rip), %xmm0          # s = 0.0
    xorl %esi, %esi                   # i = 0
.Lfi:
    movsd 0(%r9,%rsi,8), %xmm1
    movsd 0(%r12,%rsi,8), %xmm2
    mulsd %xmm2, %xmm1
    addsd %xmm1, %xmm0
    incl %esi
    cmpl %ecx, %esi
    jne .Lfi
    movsd 0(%r15,%rax,8), %xmm2       # b = Bl[j]
    addsd %xmm2, %xmm0                 # s += Bl[j]
    cmpl %r10d, %ebx
    je .Lfsig
    xorpd %xmm1, %xmm1
    ucomisd %xmm1, %xmm0              # v - 0.0
    ja .Lfkeep                        # v > +0.0
    movsd .LC_0(%rip), %xmm0          # else +0.0
.Lfkeep:
    jmp .Lfa
.Lfsig:
    movl %eax, 32(%rsp)               # j   (exp clobbers all volatiles)
    movq %r8, 40(%rsp)                # a_out
    movl %ecx, 48(%rsp)               # f
    movl %edx, 56(%rsp)               # o
    movq %r10, 64(%rsp)               # L-1
    movq %r11, 72(%rsp)               # Wl
    movsd .LC_0(%rip), %xmm1
    subsd %xmm0, %xmm1                # -v
    movsd %xmm1, %xmm0
    call exp                          # xmm0 = exp(-v)
    movl 32(%rsp), %eax               # j
    movq 40(%rsp), %r8                # a_out
    movl 48(%rsp), %ecx               # f
    movl 56(%rsp), %edx               # o
    movq 64(%rsp), %r10               # L-1
    movq 72(%rsp), %r11               # Wl
    addsd .LC_1(%rip), %xmm0          # 1 + e
    movsd .LC_1(%rip), %xmm1
    divsd %xmm0, %xmm1                # 1/(1+e)
    movsd %xmm1, %xmm0                # sigmoid result -> xmm0 (shared store)
.Lfa:
    movsd %xmm0, 0(%r8,%rax,8)
    incl %eax
    cmpl %edx, %eax
    jne .Lfj
    incl %ebx
    cmpl %r10d, %ebx
    jle .Lfl                           # run layers 0..L-1 (ebx ends at L)
    movq 0(%r13,%rbx,8), %rax         # y = A[L] (value)
    addq $88, %rsp
    popq %rsi
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# wtot/btot helper: net in rcx -> eax = wtot (doubles), edx = btot (doubles)
# ---------------------------------------------------------------------------
    .p2align 4
.Ltot2:
    pushq %rbx
    pushq %r9
    pushq %r10
    pushq %r11
    pushq %rsi
    movq %rcx, %r11
    movl NET_N(%r11), %ebx
    xorl %eax, %eax
    xorl %edx, %edx
    xorl %r9d, %r9d
.Lt2:
    movl NET_SIZES(%r11,%r9,4), %r10d
    movl %r9d, %ecx
    incl %ecx
    movl NET_SIZES(%r11,%rcx,4), %esi
    imull %esi, %r10d
    addl %r10d, %eax
    addl %esi, %edx
    incl %r9d
    cmpl %ebx, %r9d
    jne .Lt2
    popq %rsi
    popq %r11
    popq %r10
    popq %r9
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# nn_create(int L, const int *sizes, uint64 seed, const double *lr) -> Net* (rax)
# ---------------------------------------------------------------------------
    .p2align 4
    .globl nn_create
nn_create:
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    pushq %rsi
    pushq %rdi
    subq $64, %rsp              # 32 shadow + seed/btot scratch
    movl %ecx, %r15d                  # L
    movq %rdx, %r12                   # sizes
    xorl %ebx, %ebx                   # wtot
    xorl %r14d, %r14d                 # btot
    movl (%r12), %r13d                # atot = sizes[0]
    xorl %eax, %eax
.Lcs:
    movl (%r12,%rax,4), %ecx          # f
    movl 4(%r12,%rax,4), %edx         # o
    imull %edx, %ecx                  # f*o
    addl %ecx, %ebx
    addl %edx, %r14d
    addl %edx, %r13d
    incl %eax
    cmpl %r15d, %eax
    jne .Lcs
    addl %ebx, %r13d                  # atot + wtot
    movl %r14d, 48(%rsp)             # save btot (edx clobbered by malloc)
    addl %r14d, %r13d                 # total doubles
    movl %r13d, %eax
    imull $8, %eax
    addl $192, %eax                   # 192-byte header + data
    movl %eax, %ecx
    movsd (%r9), %xmm0                # lr = *lr
    movsd %xmm0, 32(%rsp)            # save lr (16-aligned slot)
    movq %r8, 40(%rsp)               # save seed
    call malloc
    testq %rax, %rax
    jz .Lcfail
    movq %rax, %r14                   # net (whole allocation)
    movq 40(%rsp), %rax
    movl %r15d, NET_N(%r14)
    movsd 32(%rsp), %xmm0
    movsd %xmm0, NET_LR(%r14)
    movq $0, NET_RNG(%r14)
    leaq 192(%r14), %r10             # data base
    movq %r10, NET_BUF(%r14)
    movl %r13d, NET_SIZE(%r14)
    leaq NET_SIZES(%r14), %rdi
    leaq (%r12), %rsi
    movl %r15d, %ecx
    incl %ecx
    rep movsl                         # copy L+1 sizes
    movq %r10, %rdi
    movl %r13d, %ecx
    xorl %eax, %eax                      # RAX = 0.0 for rep stosq (zeroing data)
    rep stosq                         # zero data region
    movl %ebx, %ecx
    imull $8, %ecx
    leaq 0(%r10,%rcx), %r11           # B cursor = data + wtot*8
    movl %ebx, %ecx
    addl 48(%rsp), %ecx               # + btot
    imull $8, %ecx
    leaq 0(%r10,%rcx), %r13           # A cursor = data + (wtot+btot)*8
    xorl %eax, %eax
.Lcp:
    movl NET_SIZES(%r14,%rax,4), %ecx # f
    movl %eax, %esi
    incl %esi
    movl NET_SIZES(%r14,%rsi,4), %edx # o
    movq %r10, NET_W(%r14,%rax,8)
    movq %r11, NET_B(%r14,%rax,8)
    movq %r13, NET_A(%r14,%rax,8)
    imull %edx, %ecx                  # f*o
    imull $8, %ecx
    addq %rcx, %r10
    movl %edx, %esi
    imull $8, %esi
    addq %rsi, %r11
    movl NET_SIZES(%r14,%rax,4), %esi # f
    imull $8, %esi
    addq %rsi, %r13
    incl %eax
    cmpl %r15d, %eax
    jne .Lcp
    movq %r13, NET_A(%r14,%rax,8)     # A[L]
    # --- rng seed: splitmix64(seed) finalizer, 0 -> 1 ---
    # mulq idiom: rax=value, rdx=K; mulq %rdx -> rax = (value*K) mod 2^64,
    # rdx = high half (unused). NOTE: "mulq %rax" would square the value.
    movq 40(%rsp), %rax
    movq .LK_SM(%rip), %rcx
    addq %rcx, %rax                      # rax = z = seed + SM
    movq %rax, %rcx
    shrq $30, %rcx
    xorq %rax, %rcx                      # rcx = z ^ (z>>30)
    movq %rcx, %rax
    movq .LK_K1(%rip), %rdx
    mulq %rdx                            # rax = (zA*K1) mod 2^64
    movq %rax, %rcx                      # z1
    movq %rcx, %rax
    shrq $27, %rax
    xorq %rcx, %rax                      # rax = z1 ^ (z1>>27)
    movq .LK_K2(%rip), %rdx
    mulq %rdx                            # rax = (zA2*K2) mod 2^64
    movq %rax, %rcx                      # z2
    movq %rcx, %rax
    shrq $31, %rax
    xorq %rcx, %rax                      # rax = z2 ^ (z2>>31)
    testq %rax, %rax
    jnz .Ls2
    movq $1, %rax
.Ls2:
    movq %rax, NET_RNG(%r14)
    # --- He-uniform init ---
    leaq NET_SIZES(%r14), %r11
    xorl %r13d, %r13d                 # l = 0
.Lil:
    movl (%r11,%r13,4), %ecx          # f
    movl 4(%r11,%r13,4), %edx         # o
    movq NET_W(%r14,%r13,8), %r10     # Wl
    movl %ecx, %r8d
    addl %edx, %r8d
    cvtsi2sdq %r8, %xmm0              # f+o
    movsd .LC_6(%rip), %xmm1
    divsd %xmm0, %xmm1                # 6.0/(f+o)
    sqrtsd %xmm1, %xmm1               # lim
    xorl %r9d, %r9d                   # j
.Lij:
    xorl %r8d, %r8d                   # i
.Lii:
    movq NET_RNG(%r14), %rcx
    movq %rcx, %rdx
    shrq $12, %rdx
    xorq %rdx, %rcx
    movq %rcx, %rdx
    shlq $25, %rdx
    xorq %rdx, %rcx
    movq %rcx, %rdx
    shrq $27, %rdx
    xorq %rdx, %rcx
    movq %rcx, NET_RNG(%r14)
    movq %rcx, %rax                  # rax = s
    movq .LK_GLD(%rip), %rdx         # rdx = GOLDEN
    mulq %rdx                        # rax = (s*GOLDEN) mod 2^64
    shrq $11, %rax                   # top 53 bits
    movl (%r11,%r13,4), %ecx          # reload f (clobbered by rng)
    movl 4(%r11,%r13,4), %edx         # reload o (clobbered by rng)
    cvtsi2sdq %rax, %xmm0
    mulsd .LC_2m53(%rip), %xmm0       # u01
    mulsd .LC_2(%rip), %xmm0
    subsd .LC_1(%rip), %xmm0
    mulsd %xmm1, %xmm0                # lim * (2*u01 - 1)
    movsd %xmm0, (%r10,%r8,8)
    incl %r8d
    cmpl %ecx, %r8d
    jne .Lii
    leaq 0(%r10,%rcx,8), %r10
    incl %r9d
    cmpl %edx, %r9d
    jne .Lij
    incl %r13d
    cmpl %r15d, %r13d
    jne .Lil
    movq %r14, %rax
    addq $64, %rsp
    popq %rdi
    popq %rsi
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    ret
.Lcfail:
    xorl %eax, %eax
    addq $64, %rsp
    popq %rdi
    popq %rsi
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# nn_free(Net*)
# ---------------------------------------------------------------------------
    .p2align 4
    .globl nn_free
nn_free:
    pushq %rbx
    subq $32, %rsp
    movq %rcx, %rbx
    testq %rbx, %rbx
    jz .Lnf1
    movq %rbx, %rcx
    call free
.Lnf1:
    addq $32, %rsp
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# nn_forward(const Net*, const double *x, double *y)
# ---------------------------------------------------------------------------
    .p2align 4
    .globl nn_forward
nn_forward:
    pushq %rbx
    pushq %rsi
    pushq %rdi
    subq $32, %rsp
    movq %rcx, %rbx
    movq %r8, %rdi                    # dst = y (rdi survives .Ldo_forward)
    call .Ldo_forward
    movq %rax, %rsi                   # src = A[L]
    movl NET_N(%rbx), %edx
    movl NET_SIZES(%rbx,%rdx,4), %ecx
    rep movsq
    addq $32, %rsp
    popq %rdi
    popq %rsi
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# nn_loss(const Net*, const double *x, const double *t) -> double in xmm0
# ---------------------------------------------------------------------------
    .p2align 4
    .globl nn_loss
nn_loss:
    pushq %rbx
    pushq %r12
    pushq %rdi
    subq $48, %rsp
    movq %rcx, %rbx
    movq %r8, %r12                    # t
    movq %rbx, %rcx                   # (rdx = x)
    call .Ldo_forward
    movq %rax, %rdi                   # y
    movl NET_N(%rbx), %ecx
    movl NET_SIZES(%rbx,%rcx,4), %edx
    xorl %ecx, %ecx
    movsd .LC_0(%rip), %xmm0
.Lls:
    movsd 0(%rdi,%rcx,8), %xmm1
    movsd 0(%r12,%rcx,8), %xmm2       # t
    subsd %xmm2, %xmm1                 # d = y - t
    mulsd %xmm1, %xmm1
    addsd %xmm1, %xmm0
    incl %ecx
    cmpl %edx, %ecx
    jne .Lls
    mulsd .LC_05(%rip), %xmm0
    addq $48, %rsp
    popq %rdi
    popq %r12
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# nn_backward(const Net*, const double *x, const double *t,
#             double *dW, double *dB)
# dW/dB are flat buffers (wtot / btot doubles), zeroed here first.
# ---------------------------------------------------------------------------
    .p2align 4
    .globl nn_backward
nn_backward:
    pushq %rbx
    pushq %r10
    pushq %r11
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    pushq %rsi
    pushq %rdi
    subq $48, %rsp
    movq 160(%rsp), %r12             # dB flat (5th arg: caller [rsp+32]; +8 retaddr, +72 pushes, +48 sub)
    movq %rcx, %r14
    movq %r8, %r15                    # t
    movq %r9, %rbx                    # dW flat
    movq %rdx, 40(%rsp)               # save x (.Lbz1 clobbers rdx)
    # --- zero dW/dB ---
    movl NET_N(%r14), %eax
    xorl %r10d, %r10d
    xorl %r11d, %r11d
    xorl %ecx, %ecx
.Lbz1:
    movl NET_SIZES(%r14,%rcx,4), %edx
    movl %ecx, %esi
    incl %esi
    movl NET_SIZES(%r14,%rsi,4), %r8d
    imull %r8d, %edx
    addl %edx, %r10d
    addl %r8d, %r11d
    incl %ecx
    cmpl %eax, %ecx
    jne .Lbz1
    movq %rbx, %rdi
    movl %r10d, %ecx
    xorl %eax, %eax                      # RAX = 0.0 for rep stosq (zeroing dW/dB)
    rep stosq
    movq %r12, %rdi
    movl %r11d, %ecx
    rep stosq
    movq %r12, %rdi                   # dB base (rdi)
    # --- forward pass ---
    movq 40(%rsp), %rdx                # restore x (.Lbz1 clobbered rdx)
    movq %r14, %rcx
    call .Ldo_forward
    movq %rax, %r12                   # y
    # --- delta_out[j] = (y-t) * y * (1-y) ---
    movl NET_N(%r14), %ecx
    movl NET_SIZES(%r14,%rcx,4), %edx
    leaq .Ldout(%rip), %r13
    xorl %ecx, %ecx
.Lbo:
    movsd 0(%r12,%rcx,8), %xmm0        # y
    movsd 0(%r15,%rcx,8), %xmm1        # t
    subsd %xmm1, %xmm0                  # y - t
    movsd 0(%r12,%rcx,8), %xmm1        # y
    mulsd %xmm1, %xmm0                  # (y-t) * y
    movsd 0(%r12,%rcx,8), %xmm1        # y
    movsd .LC_1(%rip), %xmm2
    subsd %xmm1, %xmm2                  # 1 - y
    mulsd %xmm2, %xmm0
    movsd %xmm0, 0(%r13,%rcx,8)
    incl %ecx
    cmpl %edx, %ecx
    jne .Lbo
    # --- layers l = L-1 .. 0 ---
    movl NET_N(%r14), %r10d
    decl %r10d
    leaq .Ldelt(%rip), %r8
.Lbl:
    movl %r10d, %esi
    incl %esi
    movl NET_SIZES(%r14,%r10,4), %ecx   # f
    movl NET_SIZES(%r14,%rsi,4), %edx    # o
    # dW layer base: offw = sum_{k=0..l-1} sizes[k]*sizes[k+1]
    xorl %r9d, %r9d
    xorl %eax, %eax
.Loffw:
    cmpl %r10d, %r9d                   # k >= l -> done (also l=0)
    jae .Loffw_end
    movl %r9d, %esi
    incl %esi
    movl NET_SIZES(%r14,%r9,4), %r8d
    movl NET_SIZES(%r14,%rsi,4), %r11d
    imull %r11d, %r8d
    addl %r8d, %eax
    incl %r9d
    jmp .Loffw
.Loffw_end:
    imull $8, %eax
    leaq 0(%rbx,%rax), %r15            # dW_l
    # dB layer base: offb = sum_{k=0..l-1} sizes[k+1]
    xorl %r9d, %r9d
    xorl %eax, %eax
    movl %r10d, %esi                   # bound = l
.Loffb:
    cmpl %esi, %r9d
    jae .Loffb_end
    movl %r9d, %r11d
    incl %r11d
    movl NET_SIZES(%r14,%r11,4), %r8d   # sizes[k+1]  (r11 free: Wl loaded after)
    addl %r8d, %eax
    incl %r9d
    jmp .Loffb
.Loffb_end:
    imull $8, %eax
    leaq 0(%rdi,%rax), %r9            # dB_l = dB + offset
    movq NET_W(%r14,%r10,8), %r11     # Wl
    movq NET_A(%r14,%r10,8), %r12     # a_in
    xorl %eax, %eax
.Lbj:
    movsd 0(%r13,%rax,8), %xmm1        # d = delta[j]
    movsd 0(%r9,%rax,8), %xmm0
    addsd %xmm1, %xmm0                  # dB[j] += d
    movsd %xmm0, 0(%r9,%rax,8)
    xorpd %xmm0, %xmm0
    ucomisd %xmm1, %xmm0               # 0.0 - d
    jp .Lbproc                         # NaN -> process
    jz .Lbnext                         # +/-0.0 -> skip dW
.Lbproc:
    movl %eax, %esi
    imull %ecx, %esi
    leaq 0(%r15,%rsi,8), %r8           # dW row
    xorl %esi, %esi
.Lbpi:
    movsd 0(%r8,%rsi,8), %xmm0
    movsd 0(%r12,%rsi,8), %xmm2        # a_in[i]
    mulsd %xmm1, %xmm2                  # xmm2 = d * a_in[i] (keep d in xmm1 for next i)
    addsd %xmm2, %xmm0
    movsd %xmm0, 0(%r8,%rsi,8)
    incl %esi
    cmpl %ecx, %esi
    jne .Lbpi
.Lbnext:
    incl %eax
    cmpl %edx, %eax
    jne .Lbj
    testl %r10d, %r10d
    jg .Lbdin
    jmp .Lbend
.Lbdin:
    leaq .Ldout(%rip), %rax
    cmpq %rax, %r13
    je .Lbnew1
    leaq .Ldout(%rip), %r8
    jmp .Lbnew2
.Lbnew1:
    leaq .Ldelt(%rip), %r8
.Lbnew2:
    xorl %esi, %esi
.Lbdi:
    movsd 0(%r12,%rsi,8), %xmm0        # a_prev[i]
    xorpd %xmm1, %xmm1
    ucomisd %xmm1, %xmm0               # a - 0.0
    ja .Lbdi_s
    movsd .LC_0(%rip), %xmm0
    movsd %xmm0, 0(%r8,%rsi,8)         # +0.0
    jmp .Lbdi_n
.Lbdi_s:
    leaq 0(%r11,%rsi,8), %r9           # column cursor
    movsd .LC_0(%rip), %xmm1
    xorl %eax, %eax
.Lbdi_j:
    movsd 0(%r9), %xmm0
    movsd 0(%r13,%rax,8), %xmm2
    mulsd %xmm2, %xmm0
    addsd %xmm0, %xmm1
    leaq 0(%r9,%rcx,8), %r9
    incl %eax
    cmpl %edx, %eax
    jne .Lbdi_j
    movsd %xmm1, 0(%r8,%rsi,8)
.Lbdi_n:
    incl %esi
    cmpl %ecx, %esi
    jne .Lbdi
    movq %r8, %r13
    decl %r10d
    jmp .Lbl
.Lbend:
    addq $48, %rsp
    popq %rdi
    popq %rsi
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %r11
    popq %r10
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# nn_sgd_update(Net*, const double *dW, const double *dB)
# ---------------------------------------------------------------------------
    .p2align 4
    .globl nn_sgd_update
nn_sgd_update:
    pushq %rbx
    pushq %r9
    pushq %r10
    pushq %r11
    pushq %r12
    pushq %rsi
    movq %rcx, %r12
    movq %r8, %r11                    # dB
    movl NET_N(%r12), %ebx
    xorl %r10d, %r10d
    xorl %esi, %esi
.Lsu1:
    movl NET_SIZES(%r12,%rsi,4), %ecx
    movl %esi, %r8d
    incl %r8d
    movl NET_SIZES(%r12,%r8,4), %r9d
    imull %r9d, %ecx
    addl %ecx, %r10d
    incl %esi
    cmpl %ebx, %esi
    jne .Lsu1
    movq NET_BUF(%r12), %rbx
    movsd NET_LR(%r12), %xmm1
    movq %rdx, %r9                    # dW
    xorl %ecx, %ecx
.Lsuw:
    movsd 0(%r9,%rcx,8), %xmm0
    mulsd %xmm1, %xmm0
    movsd 0(%rbx,%rcx,8), %xmm2
    subsd %xmm0, %xmm2
    movsd %xmm2, 0(%rbx,%rcx,8)
    incl %ecx
    cmpl %r10d, %ecx
    jne .Lsuw
    movl NET_N(%r12), %ebx
    xorl %esi, %esi
    xorl %r9d, %r9d
.Lsu2:
    movl %esi, %r8d
    incl %r8d
    movl NET_SIZES(%r12,%r8,4), %ecx
    addl %ecx, %r9d
    incl %esi
    cmpl %ebx, %esi
    jne .Lsu2
    movq NET_BUF(%r12), %rbx           # reload data base (movl %ebx clobbered rbx)
    movl %r10d, %ecx
    imull $8, %ecx
    leaq 0(%rbx,%rcx), %r10           # B ptr
    xorl %ecx, %ecx
.Lsub:
    movsd 0(%r11,%rcx,8), %xmm0
    mulsd %xmm1, %xmm0
    movsd 0(%r10,%rcx,8), %xmm2
    subsd %xmm0, %xmm2
    movsd %xmm2, 0(%r10,%rcx,8)
    incl %ecx
    cmpl %r9d, %ecx
    jne .Lsub
    popq %rsi
    popq %r12
    popq %r11
    popq %r10
    popq %r9
    popq %rbx
    ret

# ---------------------------------------------------------------------------
# nn_get_weights(const Net*, double *W, double *B)
# nn_set_weights(Net*, const double *W, const double *B)
# flat copies; W is wtot doubles, B is btot doubles.
# ---------------------------------------------------------------------------
    .p2align 4
    .globl nn_get_weights
nn_get_weights:
    pushq %rbx
    pushq %r10
    pushq %r11
    pushq %rsi
    pushq %rdi
    subq $32, %rsp
    movq %rcx, %rbx
    movq %rdx, %r10           # W dst (rdx clobbered by .Ltot2)
    movq %r8, %r11            # B dst
    call .Ltot2               # eax = wtot, edx = btot
    movq NET_BUF(%rbx), %rsi
    movq %r10, %rdi
    movl %eax, %ecx
    rep movsq
    movl %eax, %ecx
    imull $8, %ecx
    movq NET_BUF(%rbx), %rsi             # reload data base (rsi advanced by W copy)
    leaq 0(%rsi,%rcx), %rsi              # B base = data + wtot*8
    movq %r11, %rdi
    movl %edx, %ecx
    rep movsq
    addq $32, %rsp
    popq %rdi
    popq %rsi
    popq %r11
    popq %r10
    popq %rbx
    ret

    .p2align 4
    .globl nn_set_weights
nn_set_weights:
    pushq %rbx
    pushq %r10
    pushq %r11
    pushq %rsi
    pushq %rdi
    subq $32, %rsp
    movq %rcx, %rbx
    movq %rdx, %r10           # W src
    movq %r8, %r11            # B src
    call .Ltot2               # eax = wtot, edx = btot
    movq NET_BUF(%rbx), %rdi
    movq %r10, %rsi
    movl %eax, %ecx
    rep movsq
    movl %eax, %ecx
    imull $8, %ecx
    movq NET_BUF(%rbx), %rdi             # reload data base (rdi advanced by W copy)
    leaq 0(%rdi,%rcx), %rdi              # B base = data + wtot*8
    movq %r11, %rsi
    movl %edx, %ecx
    rep movsq
    addq $32, %rsp
    popq %rdi
    popq %rsi
    popq %r11
    popq %r10
    popq %rbx
    ret

# TEMP DEBUG: nn_dbg_act(net, int l, double *out, int n): copy A[l] -> out
    .p2align 4
    .globl nn_dbg_act
nn_dbg_act:
    pushq %rdi
    leaq NET_A(%rcx,%rdx,8), %rax
    movq %r8, %rdi                 # dst = out
    movq (%rax), %rsi              # src = A[l]
    movl %r9d, %ecx
    rep movsq
    popq %rdi
    ret

    .bss
    .align 16
.Ldout:
    .skip 256 * 8                     # output-delta scratch (<=256 units)
.Ldelt:
    .skip 256 * 8                     # hidden-delta scratch (double buffer)
