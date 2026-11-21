"""讲义：第 22 章 · FlashAttention 1–4（notes/lessons/22-flashattention.md）

运行：uv run python ch22_flashattention.py
用 NumPy 写一遍 FA2 风格的分块前向（online softmax 连同 O 一起打折、最后除一次 ℓ），和标准 attention 对拍；
再算中间矩阵多大、FA4 的 feeds and speeds。真正的 kernel 要 GPU [GPU]。
"""

import numpy as np

MB = 2**20


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """标准写法：S = QKᵀ（N × N 整张落地）→ P = softmax(S) → O = PV。这里不除 √d，和讲义的公式一致。"""
    S = Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V


def flash_forward(Q, K, V, br: int, bc: int) -> tuple[np.ndarray, np.ndarray]:
    """FA2 前向：每个 Q 块（br 行）扫过所有 K、V 块（bc 行）；S 只在片上存在，用完即丢。返回 O 和 LSE。"""
    N = Q.shape[0]
    O = np.zeros((N, V.shape[1]))
    LSE = np.zeros(N)
    for i in range(0, N, br):
        q = Q[i:i + br]
        o = np.zeros((len(q), V.shape[1]))
        m = np.full(len(q), -np.inf)
        l = np.zeros(len(q))
        for j in range(0, N, bc):
            S = q @ K[j:j + bc].T                         # br × bc，留在片上
            m_new = np.maximum(m, S.max(axis=1))
            P = np.exp(S - m_new[:, None])
            scale = np.exp(m - m_new)                      # 旧账按新基准打折
            l = scale * l + P.sum(axis=1)
            o = scale[:, None] * o + P @ V[j:j + bc]       # O 也一起打折
            m = m_new
        O[i:i + br] = o / l[:, None]                       # 最后统一除一次 ℓ
        LSE[i:i + br] = m + np.log(l)
    return O, LSE


def sizes(n: int, d: int, bytes_per: int = 2) -> tuple[float, float]:
    """一个头：Q / K / V / O 各多大、S / P 各多大（MB）。"""
    return n * d * bytes_per / MB, n * n * bytes_per / MB


def fa4_cycles(m: int, n: int, d: int, mma_flops: int = 8192, exp_per_cycle: int = 16) -> tuple[float, float]:
    """FA4 §3.1.1：一个 tile 里两次 GEMM 共 4MNd FLOP，Tensor Core 每 SM 每周期 8192；exp 每 SM 每周期 16 次。"""
    return 4 * m * n * d / mma_flops, m * n / exp_per_cycle


def main():
    print("\n第一步 N = 8192、d = 128、BF16，一个头")
    qkv, sp = sizes(8192, 128)
    print(f"  Q、K、V、O 各 {qkv:.0f} MB；S、P 各 {sp:.0f} MB；大 {sp / qkv:.0f} 倍")

    print("\n第二步 分块前向 vs 标准 attention（N = 64、d = 16、块 16 × 16）")
    rng = np.random.default_rng(0)
    Q, K, V = (rng.standard_normal((64, 16)) for _ in range(3))
    O, LSE = flash_forward(Q, K, V, 16, 16)
    print("  O 一致（不是近似）：", np.allclose(O, attention(Q, K, V)))
    S = Q @ K.T
    print("  LSE = m + log ℓ 等于 log Σ e^S：", np.allclose(LSE, np.log(np.exp(S).sum(axis=1))))
    print("  换成块 8 × 32 结果也一样：", np.allclose(flash_forward(Q, K, V, 8, 32)[0], O))
    print(f"  片上一次只放 16 × 16 的 S（{16 * 16} 个数），标准写法要放 64 × 64（{64 * 64} 个）")

    print("\n第三步 A100：FP16 矩阵乘 312 TFLOP/s ÷ 非矩阵乘 FP32 19.5 TFLOP/s =", 312 / 19.5)

    print("\n第五步 FA4 的 feeds and speeds，tile M = N = d = 128")
    t_mma, t_exp = fa4_cycles(128, 128, 128)
    print(f"  T_MMA = {t_mma:.0f} 周期，T_exp = {t_exp:.0f} 周期")


if __name__ == "__main__":
    main()
