"""讲义：第 21 章 · online softmax 与算子融合（notes/lessons/21-online-softmax-fusion.md）

运行：uv run python ch21_online_softmax.py
减最大值防溢出；边读边维护 (m, ℓ)；两块的 (m, ℓ) 合并；最后算一笔融合省下的显存账。
"""

import math

import numpy as np


def softmax_naive(x: np.ndarray) -> np.ndarray:
    """直接 e^x / Σe^x：数大了会溢出。"""
    e = np.exp(x)
    return e / e.sum()


def softmax_safe(x: np.ndarray) -> np.ndarray:
    """先减最大值：所有指数 ≤ 0，不会溢出；结果不变。"""
    e = np.exp(x - x.max())
    return e / e.sum()


def online_ml(xs) -> list[tuple[float, float]]:
    """一遍扫过去维护 (m, ℓ)：遇到更大的数，旧账乘 e^(m_old − m_new)。返回每读一个数之后的 (m, ℓ)。"""
    m, l = -math.inf, 0.0
    trace = []
    for x in xs:
        m_new = max(m, x)
        l = l * math.exp(m - m_new) + math.exp(x - m_new)   # m = −∞ 时 exp(−∞) = 0
        m = m_new
        trace.append((m, l))
    return trace


def merge(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """两块的 (m, ℓ) 合并：m = max(m1, m2)，ℓ = ℓ1 e^(m1 − m) + ℓ2 e^(m2 − m)。"""
    m = max(a[0], b[0])
    return m, a[1] * math.exp(a[0] - m) + b[1] * math.exp(b[0] - m)


def softmax_two_pass(x: np.ndarray) -> np.ndarray:
    """online 版：第一遍得到 (m, ℓ)，第二遍输出 e^(x − m) / ℓ。"""
    m, l = online_ml(x)[-1]
    return np.exp(x - m) / l


def fusion_bytes(n: int, bytes_per: int = 2) -> tuple[int, int]:
    """y = gelu(x + bias)：两个 kernel 读 x、写 t、读 t、写 y；融合后只读 x、写 y。"""
    return 4 * n * bytes_per, 2 * n * bytes_per


def main():
    print("\n第一步 e^100 放不进 FP16（最大约 65504）")
    with np.errstate(over="ignore", invalid="ignore"):
        print("  np.exp(float16(100)) =", np.exp(np.float16(100)), "；float32:", np.exp(np.float32(100)))
        x = np.array([100, 101, 102], dtype=np.float16)
        print("  直接算 softmax：", softmax_naive(x), "；先减最大值：", softmax_safe(x))

    print("\n第三步 online softmax，x = [1, 3, 2]")
    for v, (m, l) in zip([1, 3, 2], online_ml([1, 3, 2])):
        print(f"  读到 {v}: m = {m}, ℓ = {l:.4f}")
    direct = sum(math.exp(v - 3) for v in [1, 3, 2])
    print(f"  直接算 Σ e^(x − 3) = {direct:.4f}")
    xs = np.array([1.0, 3.0, 2.0])
    print("  两遍版 softmax 和三遍版一样：", np.allclose(softmax_two_pass(xs), softmax_safe(xs)))

    print("\n第三步 分块合并：[1, 3] 和 [2] 各算各的，再合并")
    print("  ", merge(online_ml([1, 3])[-1], online_ml([2])[-1]), "= 一次读完的", online_ml([1, 3, 2])[-1])

    print("\n第二、三步 访存次数：三遍读 + 一遍写 = 4 次；online 两遍读 + 一遍写 = 3 次")

    print("\n第四步 1 亿个 BF16 做 y = gelu(x + bias)")
    two, fused = fusion_bytes(100_000_000)
    print(f"  两个 kernel：{two / 1e6:.0f} MB；融合：{fused / 1e6:.0f} MB；耗时约为 {fused / two:.0%}")


if __name__ == "__main__":
    main()
