"""讲义：第 20 章 · Blackwell：tcgen05、TMEM、块缩放（notes/lessons/20-blackwell-tcgen05.md）

运行：uv run python ch20_tcgen05_tmem.py
算 TMEM 的大小、按 warp 分 lane；再用 NumPy 模拟 FP4（E2M1）的分块缩放，比较 MXFP4 和 NVFP4 的误差。
tcgen05 / TMEM 要 sm_100a [GPU]；这里的量化是"先缩放、再取最近的 FP4 值"的数学模拟，不是硬件实现。
"""

import math

import numpy as np

TMEM_LANES, TMEM_COLS, CELL_BYTES = 128, 512, 4
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])   # FP4 E2M1 能表示的非负值；加上符号一共 16 个编码


def tmem_bytes() -> int:
    """每个 CTA 的 TMEM：128 行 × 512 列 × 每格 32 位。"""
    return TMEM_LANES * TMEM_COLS * CELL_BYTES


def tmem_alloc_cols(cols_needed: int) -> int:
    """tcgen05.alloc 以列为单位：至少 32 列、必须是 2 的幂。"""
    return max(32, 1 << math.ceil(math.log2(cols_needed)))


def warp_lanes(warp: int) -> range:
    """每个 warp 只能访问 TMEM 的 1/4：warp w 管第 32w .. 32w + 31 行（lane）。"""
    return range(32 * (warp % 4), 32 * (warp % 4) + 32)


def round_e4m3(v: np.ndarray) -> np.ndarray:
    """把正数舍入到 FP8 E4M3（3 位尾数、最大 448）：用来存 NVFP4 的缩放因子。"""
    e = np.maximum(np.floor(np.log2(v)), -6)
    q = 2.0 ** (e - 3)
    return np.minimum(np.round(v / q) * q, 448)


def round_ue8m0(v: np.ndarray) -> np.ndarray:
    """UE8M0：只有指数、没有尾数，缩放因子只能是 2 的幂（向上取，保证不溢出）。"""
    return 2.0 ** np.ceil(np.log2(v))


def to_e2m1(y: np.ndarray) -> np.ndarray:
    """取最近的 E2M1 值（带符号，超出 ±6 的截到 ±6）。"""
    mag = np.abs(y)
    idx = np.abs(mag[..., None] - E2M1).argmin(axis=-1)
    return np.sign(y) * E2M1[idx]


def block_quantize(x: np.ndarray, block: int, scale_fmt: str) -> np.ndarray:
    """分块缩放：每 block 个元素共享一个 scale，x ≈ scale × q。返回反量化后的值（用来算误差）。"""
    xb = x.reshape(-1, block)
    amax = np.abs(xb).max(axis=1, keepdims=True)
    raw = np.where(amax > 0, amax / 6, 1.0)                 # 让块里最大的数落在 FP4 的最大值 6 附近
    scale = round_ue8m0(raw) if scale_fmt == "ue8m0" else round_e4m3(raw)
    return (to_e2m1(xb / scale) * scale).ravel()


def main():
    print("\n第二步 TMEM")
    print(f"  128 行 × 512 列 × 4 字节 = {tmem_bytes()} 字节 = {tmem_bytes() // 1024} KB / SM")
    print("  申请 20 列 →", tmem_alloc_cols(20), "列；申请 100 列 →", tmem_alloc_cols(100), "列")
    print("  4 个 warp 各管的 lane：", [(r.start, r.stop - 1) for r in (warp_lanes(w) for w in range(4))])

    print("\n第四步 2-SM MMA：单 SM 的 M 最大 128，CTA pair 拼起来 M =", 128 * 2, "，每个 SM 只放一半的 B")

    print("\n第五步 FP4（E2M1）")
    print("  非负值：", E2M1, "；加上负号一共", 2 * len(E2M1), "个编码（含 ±0）")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096) * np.exp(rng.standard_normal(4096))   # 编的数据：大小差得很多
    one_scale = to_e2m1(x / (np.abs(x).max() / 6)) * (np.abs(x).max() / 6)
    mx = block_quantize(x, 32, "ue8m0")
    nv = block_quantize(x, 16, "e4m3")
    err = lambda y: np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(x ** 2))
    print(f"  整个张量一个 scale 的相对误差：{err(one_scale):.3f}")
    print(f"  MXFP4（32 个一块、2 的幂 scale）：{err(mx):.3f}")
    print(f"  NVFP4（16 个一块、E4M3 scale）  ：{err(nv):.3f}")


if __name__ == "__main__":
    main()
