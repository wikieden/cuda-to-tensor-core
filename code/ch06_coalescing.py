"""讲义：第 6 章 · 合并访存（notes/lessons/06-coalescing.md）

运行：uv run python ch06_coalescing.py
给出一个 warp 的 32 个字节地址，数它们落在几个 32 字节的"箱"（sector）里。
"""

import numpy as np

SECTOR = 32  # 显存按 32 字节一箱发货
WARP = 32


def sectors(byte_addrs: np.ndarray) -> int:
    """32 个地址落在几箱：不同的 ⌊地址 / 32⌋ 的个数。"""
    return len(np.unique(np.asarray(byte_addrs) // SECTOR))


def warp_addrs(stride: int, offset: int = 0, word: int = 4) -> np.ndarray:
    """线程 t 读 a[t * stride + offset]，每个元素 word 字节；返回 32 个字节地址（数组从 0 号地址开始，已对齐）。"""
    t = np.arange(WARP)
    return (t * stride + offset) * word


def utilization(byte_addrs: np.ndarray, word: int = 4) -> float:
    """利用率 = 用到的字节 ÷ 搬来的字节。"""
    return len(byte_addrs) * word / (sectors(byte_addrs) * SECTOR)


def row_major_addrs(rows: np.ndarray, cols: np.ndarray, n_cols: int, word: int = 4) -> np.ndarray:
    """二维数组 A[r][c] 按行优先存：位置 = r * N + c。"""
    return (rows * n_cols + cols) * word


def main():
    print("\n第二步 四种情况（每个线程读一个 4 字节 float）")
    cases = [("① a[i]", 1, 0), ("② a[i + 1]", 1, 1), ("③ a[2 * i]", 2, 0), ("④ a[8 * i]", 8, 0), ("④ a[32 * i]", 32, 0)]
    for name, s, o in cases:
        ad = warp_addrs(s, o)
        print(f"  {name:<12} {sectors(ad):2d} 箱，搬 {sectors(ad) * SECTOR:4d} B，利用率 {utilization(ad):.1%}")

    print("\n第二步 跨步 s ≤ 8 时箱数 = 4s，利用率 = 1/s")
    print("  s = 1..8 的箱数：", [sectors(warp_addrs(s)) for s in range(1, 9)])

    print("\n第二步 顺序打乱没关系")
    rng = np.random.default_rng(0)
    print("  把 a[i] 的 32 个地址随机打乱后仍是", sectors(rng.permutation(warp_addrs(1))), "箱")

    print("\n第三步 二维数组 N = 64：warp 里 threadIdx.x = 0..31 对应 col，row 固定为 3")
    tx = np.arange(WARP)
    print("  in[row * N + col]（按行读）：", sectors(row_major_addrs(3, tx, 64)), "箱")
    print("  in[col * N + row]（按列读）：", sectors(row_major_addrs(tx, 3, 64)), "箱")

    print("\n第四步 float4：每个线程一次读 16 字节")
    ad = warp_addrs(1, 0, word=16)
    print(f"  一条指令搬 {WARP * 16} B，{sectors(ad)} 箱，利用率 {utilization(ad, word=16):.0%}；指令数是 1/4")


if __name__ == "__main__":
    main()
