"""讲义：第 13 章 · SGEMM ①：合并访存与 shared memory 分块（notes/lessons/13-sgemm-tiling.md）

运行：uv run python ch13_sgemm_tiling.py
Kernel 1 / 2 的线程映射用第 6 章的"数箱子"判断合并不合并；Kernel 3 用 NumPy 按 tile 一段段算，并数显存读了多少次。
"""

import numpy as np

from ch06_coalescing import sectors

BOEHM = {"1 naive": 309, "2 合并访存": 1987, "3 shared 分块": 2980, "cuBLAS": 23250}  # A6000 GFLOP/s


def kernel1_coords(tx: np.ndarray, bx: int = 0, by: int = 0, ty: int = 0, bs: int = 32):
    """Kernel 1：x（行）= blockIdx.x * 32 + threadIdx.x，y（列）= blockIdx.y * 32 + threadIdx.y。"""
    return bx * bs + tx, by * bs + ty


def kernel2_coords(tx: np.ndarray, bx: int = 0, by: int = 0, bs: int = 32):
    """Kernel 2：x = blockIdx.x * 32 + threadIdx.x / 32，y = blockIdx.y * 32 + threadIdx.x % 32。"""
    return bx * bs + tx // bs, by * bs + tx % bs


def warp_read_sectors(rows, cols, ld: int) -> int:
    """一个 warp 读 M[row * ld + col]（float）落在几箱。"""
    return sectors((np.asarray(rows) * ld + np.asarray(cols)) * 4)


def reuse_factor(m: int, n: int, k: int) -> float:
    """naive 一共读 2MNK 次，A、B 只有 MK + KN 个数：平均每个数被读几次。"""
    return 2 * m * n * k / (m * k + k * n)


def naive_gemm(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int]:
    """每个"线程"(x, y) 做一个长度 K 的点积；返回 C 和从显存读的数的个数。"""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=A.dtype)
    reads = 0
    for x in range(M):
        for y in range(N):
            C[x, y] = A[x, :] @ B[:, y]     # 读 K 个 A、K 个 B
            reads += 2 * K
    return C, reads


def tiled_gemm(A: np.ndarray, B: np.ndarray, bs: int = 32) -> tuple[np.ndarray, int]:
    """Kernel 3：每个 block 负责 C 的 bs × bs 小块，沿 K 每次搬 A、B 各一个 bs × bs 的 tile 进 shared。"""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=A.dtype)
    reads = 0
    for bx in range(0, M, bs):
        for by in range(0, N, bs):
            acc = np.zeros((bs, bs), dtype=A.dtype)
            for bk in range(0, K, bs):
                As = A[bx:bx + bs, bk:bk + bs]       # ① 搬进 shared（每人搬 1 个）
                Bs = B[bk:bk + bs, by:by + bs]
                reads += As.size + Bs.size
                acc += As @ Bs                        # ③ 在 shared 里做 bs 次乘加
            C[bx:bx + bs, by:by + bs] = acc
    return C, reads


def main():
    print("\n第一步 Kernel 1 的算术强度：每次乘加 2 FLOP、读 8 字节")
    print(f"  {2 / 8} FLOP/byte，A6000 脊点 ≈ {30e12 / 768e9:.0f}；性能 {BOEHM['1 naive'] / BOEHM['cuBLAS']:.1%} cuBLAS")

    print("\n第二步 一个 warp（threadIdx.x = 0..31）读 A[x * K + i]，K = 4096、i = 0")
    tx = np.arange(32)
    x1, _ = kernel1_coords(tx)
    x2, y2 = kernel2_coords(tx)
    print("  Kernel 1：32 个线程的 x =", x1[:4], "… → 读 A 落在", warp_read_sectors(x1, 0, 4096), "箱")
    print("  Kernel 2：x 全是", np.unique(x2), "，读 A 落在", warp_read_sectors(x2, 0, 4096), "箱（广播）")
    print("  Kernel 2：y =", y2[:4], "… 连续，读 B[i * N + y] 落在", warp_read_sectors(0, y2, 4096), "箱（合并）")
    print(f"  性能 {BOEHM['2 合并访存'] / BOEHM['cuBLAS']:.1%}，比 Kernel 1 快 {BOEHM['2 合并访存'] / BOEHM['1 naive']:.1f} 倍")

    print("\n第三步 M = N = K = 4096：每个数平均被读几次")
    print("  ", reuse_factor(4096, 4096, 4096))

    print("\n第四步 Kernel 3：64 × 64 的小矩阵，tile 32")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((64, 64)).astype(np.float32)
    B = rng.standard_normal((64, 64)).astype(np.float32)
    C1, r1 = naive_gemm(A, B)
    C3, r3 = tiled_gemm(A, B, 32)
    print("  两种写法结果一致：", np.allclose(C1, A @ B, atol=1e-4) and np.allclose(C3, A @ B, atol=1e-4))
    print(f"  显存读：naive {r1} 次，分块 {r3} 次，少了 {r1 // r3} 倍（= tile 边长）")
    print("  每 block 的 shared：2 × 32 × 32 × 4 =", 2 * 32 * 32 * 4, "字节 = 8 KB")
    print(f"  性能 {BOEHM['3 shared 分块'] / BOEHM['cuBLAS']:.1%}")


if __name__ == "__main__":
    main()
