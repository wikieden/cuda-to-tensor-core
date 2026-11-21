"""讲义：第 23 章 · 推理 kernel（notes/lessons/23-inference-kernels.md）

运行：uv run python ch23_inference_kernels.py
decode 的算术强度和速度上限；KV cache 大小；用页表读分页的 KV cache；split-KV 分段算再按 (m, ℓ) 合并。
"""

import numpy as np

from ch22_flashattention import attention


def decode_intensity(b: int, k: int, n: int, bytes_per: int = 2) -> float:
    """[b × K] × [K × N]：FLOP = 2bKN，字节 ≈ 权重 KN × 2（b 很小时激活可以忽略）。"""
    return 2 * b * k * n / (k * n * bytes_per)


def ms_per_token(weight_bytes: float, bw: float = 3.35e12) -> float:
    """每个 token 至少把全部权重读一遍：权重字节 ÷ 带宽。"""
    return weight_bytes / bw * 1e3


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per: int = 2) -> int:
    """K 和 V 两份 × 层数 × KV 头数 × 头维度 × 每元素字节。"""
    return 2 * layers * kv_heads * head_dim * bytes_per


def paged_gather(pool: np.ndarray, block_table: list[int], length: int, page: int) -> np.ndarray:
    """按页表把一条序列的 K（或 V）从显存池里取出来：token t 在第 t // page 页、页内第 t % page 个。"""
    t = np.arange(length)
    phys = np.array(block_table)[t // page] * page + t % page
    return pool[phys]


def split_kv(q: np.ndarray, K: np.ndarray, V: np.ndarray, splits: int) -> np.ndarray:
    """把 K、V 沿长度切成几段，每段算局部输出 O_s 和 (m_s, ℓ_s)，再合并：O = Σ ℓ_s e^(m_s − m) O_s / ℓ。"""
    parts = []
    for Ks, Vs in zip(np.array_split(K, splits), np.array_split(V, splits)):
        s = q @ Ks.T
        m = s.max(axis=-1)
        p = np.exp(s - m[..., None])
        parts.append((m, p.sum(axis=-1), p @ Vs / p.sum(axis=-1)[..., None]))   # 每段已经归一化的 O_s
    m = np.max([pm for pm, _, _ in parts], axis=0)
    w = [pl * np.exp(pm - m) for pm, pl, _ in parts]
    l = np.sum(w, axis=0)
    return sum(wi[..., None] * po for wi, (_, _, po) in zip(w, parts)) / l[..., None]


def main():
    print("\n第一步 decode：batch = 1 时的算术强度（K = N = 4096）")
    print("  ≈", decode_intensity(1, 4096, 4096), "FLOP/byte，H100 脊点 295")
    t = ms_per_token(16e9)
    print(f"  Llama-3-8B（16 GB BF16）：每个 token 至少 {t:.1f} ms → batch 1 上限约 {1e3 / t:.0f} token/s")

    print("\n第二步 KV cache（32 层、8 个 KV 头、头维度 128、BF16）")
    per = kv_bytes_per_token(32, 8, 128)
    print(f"  每 token {per // 1024} KB；8K token 的序列 = {per * 8192 / 2**30:.0f} GB")

    print("\n第二步 paged attention：一条 40 个 token 的序列，每页 16 个 token，用了 3 页")
    rng = np.random.default_rng(0)
    d, page = 8, 16
    K = rng.standard_normal((40, d))
    pool = rng.standard_normal((10 * page, d))      # 显存池：10 页，里面是别人的数据
    table = [7, 2, 5]                               # 这条序列的页表：逻辑第 0、1、2 页 → 物理第 7、2、5 页
    for logical, phys in enumerate(table):
        n = min(page, 40 - logical * page)
        pool[phys * page:phys * page + n] = K[logical * page:logical * page + n]
    print("  按页表取回来的 K 和连续存放的一样：", np.array_equal(paged_gather(pool, table, 40, page), K))
    print("  token 20 在逻辑第", 20 // page, "页、页内第", 20 % page, "个 → 物理位置", table[20 // page] * page + 20 % page)

    print("\n第三步 split-KV：一个 query、2000 个 token，切 4 段")
    q = rng.standard_normal((1, 64))
    K2, V2 = rng.standard_normal((2000, 64)) / 8, rng.standard_normal((2000, 64))
    print("  合并结果和一次算完一样：", np.allclose(split_kv(q, K2, V2, 4), attention(q, K2, V2)))

    print("\n第五步 INT4 权重：字节数降为 1/4")
    print(f"  16 GB → 4 GB：每个 token 至少 {ms_per_token(4e9):.1f} ms，上限提高 {ms_per_token(16e9) / ms_per_token(4e9):.0f} 倍")


if __name__ == "__main__":
    main()
