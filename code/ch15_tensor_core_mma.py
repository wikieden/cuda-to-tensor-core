"""讲义：第 15 章 · Tensor Core：WMMA、mma.sync、ldmatrix（notes/lessons/15-tensor-core-mma.md）

运行：uv run python ch15_tensor_core_mma.py
按 PTX §9.7.16.5.8 的 m16n8k16 片段布局，把 A、B、C 拆进 32 个 lane 的"寄存器"，
做一次 D = A × B + C 再拼回来，验证布局不重不漏。真正的 mma.sync 要 Tensor Core [GPU]。
"""

import numpy as np

M, N, K = 16, 8, 16
WARP = 32


def frag_a(lane: int) -> list[tuple[int, int]]:
    """A（16 × 16，BF16）：lane 拿 8 个元素（4 个 32 位寄存器 × 2）。groupID = lane >> 2，t = lane % 4。"""
    g, t = lane >> 2, lane % 4
    return [(g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1),
            (g, 2 * t + 8), (g, 2 * t + 9), (g + 8, 2 * t + 8), (g + 8, 2 * t + 9)]


def frag_b(lane: int) -> list[tuple[int, int]]:
    """B（16 × 8）：lane 拿 4 个元素（2 个 32 位寄存器）。"""
    g, t = lane >> 2, lane % 4
    return [(2 * t, g), (2 * t + 1, g), (2 * t + 8, g), (2 * t + 9, g)]


def frag_cd(lane: int) -> list[tuple[int, int]]:
    """C / D（16 × 8，FP32）：lane 拿 4 个元素（4 个 32 位寄存器）。"""
    g, t = lane >> 2, lane % 4
    return [(g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1)]


def scatter(mat: np.ndarray, frag) -> np.ndarray:
    """把一个小矩阵拆进 32 个 lane：返回形状 (32, 每 lane 元素数) 的"寄存器"表。"""
    return np.array([[mat[r, c] for r, c in frag(lane)] for lane in range(WARP)])


def gather(regs: np.ndarray, frag, shape) -> np.ndarray:
    """反过来：从 32 个 lane 的寄存器拼回小矩阵。"""
    out = np.full(shape, np.nan)
    for lane in range(WARP):
        for v, (r, c) in zip(regs[lane], frag(lane)):
            out[r, c] = v
    return out


def mma_sync(a_regs, b_regs, c_regs) -> np.ndarray:
    """模拟一条 mma.sync.m16n8k16：硬件把 32 个 lane 的片段合成矩阵，算 D = A × B + C，再按 D 的布局分回去。"""
    A = gather(a_regs, frag_a, (M, K))
    B = gather(b_regs, frag_b, (K, N))
    C = gather(c_regs, frag_cd, (M, N))
    return scatter(A @ B + C, frag_cd)


def swizzle_chunk(row: int, chunk: int) -> int:
    """第 15 章第五步的 swizzle：以 16 字节为单位，物理位置 = 列号 XOR (行号 % 8)。"""
    return chunk ^ (row % 8)


def main():
    print("\n第一步 一条 m16n8k16 做多少次乘加")
    print("  16 × 8 × 16 =", M * N * K)

    print("\n第二步 WMMA 16 × 16 的片段：每个线程", 16 * 16 // WARP, "个数")

    print("\n第三步 每个 lane 拿几个元素")
    print(f"  A: {len(frag_a(0))} 个，B: {len(frag_b(0))} 个，C/D: {len(frag_cd(0))} 个")
    for name, frag, shape in (("A", frag_a, (M, K)), ("B", frag_b, (K, N)), ("C/D", frag_cd, (M, N))):
        cover = {rc for lane in range(WARP) for rc in frag(lane)}
        print(f"  {name:<3}: 32 个 lane 合起来覆盖 {len(cover)} 个位置 = {shape[0]} × {shape[1]}，不重不漏")
    print("  lane 5 拿 A 的：", frag_a(5))

    print("\n第三步 拆进 32 个 lane → mma → 拼回来")
    rng = np.random.default_rng(0)
    A = rng.integers(-3, 4, (M, K)).astype(np.float32)   # 小整数，方便看出完全相等
    B = rng.integers(-3, 4, (K, N)).astype(np.float32)
    C = rng.integers(-3, 4, (M, N)).astype(np.float32)
    d_regs = mma_sync(scatter(A, frag_a), scatter(B, frag_b), scatter(C, frag_cd))
    print("  拼回来的 D 和 A @ B + C 完全一样：", np.array_equal(gather(d_regs, frag_cd, (M, N)), A @ B + C))
    print("  lane 0 手里的 4 个 D 值：", d_regs[0])

    print("\n第五步 swizzle：8 行里逻辑上的第 0 块（16 字节）各放到第几块")
    print("  ", [swizzle_chunk(r, 0) for r in range(8)])
    print("  每一行仍是 0–7 的一个排列：", all(sorted(swizzle_chunk(r, c) for c in range(8)) == list(range(8))
                                          for r in range(8)))


if __name__ == "__main__":
    main()
