"""讲义：第 14 章 · SGEMM ②：寄存器分块、向量化、warptiling（notes/lessons/14-sgemm-register-tiling.md）

运行：uv run python ch14_register_tiling.py
数"每步读 shared 几次、乘加几次"；再用 NumPy 按 Kernel 5 的外积循环算一个 8 × 8 的线程小块，和 A @ B 对拍。
"""

import numpy as np

CUBLAS = 23250  # A6000 GFLOP/s
BOEHM = [("1", "naive", 309), ("2", "合并访存", 1987), ("3", "shared 分块", 2980), ("4", "1D 寄存器分块", 8475),
         ("5", "2D 寄存器分块（外积）", 15972), ("6", "向量化 + 转置 As", 18237), ("9", "自动调参", 19721),
         ("10", "warptiling", 21779)]


def reads_and_fmas(tm: int, tn: int) -> tuple[int, int, float]:
    """每个线程算 TM × TN 个结果：每步读 shared TM + TN 次（1 × 1 时是 2 次），做 TM × TN 次乘加。"""
    reads = tm + tn
    fmas = tm * tn
    return reads, fmas, fmas / reads


def thread_tile_outer(As: np.ndarray, Bs: np.ndarray, row: int, col: int, tm: int = 8, tn: int = 8) -> np.ndarray:
    """Kernel 5 的内循环：对每个 dotIdx，读 A 的一小列到 regM、B 的一小行到 regN，做 TM × TN 次乘加。

    As: BM × BK（shared 里的 A tile），Bs: BK × BN；(row, col) 是线程在 block 里的位置。
    """
    acc = np.zeros((tm, tn), dtype=As.dtype)                # threadResults，放寄存器
    for dot in range(As.shape[1]):
        reg_m = As[row * tm:(row + 1) * tm, dot]            # 从 shared 读 TM 个
        reg_n = Bs[dot, col * tn:(col + 1) * tn]            # 从 shared 读 TN 个
        acc += np.outer(reg_m, reg_n)                       # TM × TN 次乘加
    return acc


def block_tile(As: np.ndarray, Bs: np.ndarray, tm: int = 8, tn: int = 8) -> np.ndarray:
    """block 里所有线程各算自己的小块，拼起来就是整个 BM × BN 的结果。"""
    bm, bn = As.shape[0], Bs.shape[1]
    out = np.zeros((bm, bn), dtype=As.dtype)
    for r in range(bm // tm):
        for c in range(bn // tn):
            out[r * tm:(r + 1) * tm, c * tn:(c + 1) * tn] = thread_tile_outer(As, Bs, r, c, tm, tn)
    return out


def main():
    print("\n第二、三步 每步读 shared 几次、乘加几次")
    for tm, tn in ((1, 1), (8, 1), (8, 8)):
        r, f, ratio = reads_and_fmas(tm, tn)
        print(f"  {tm} × {tn}: 读 {r:2d}，乘加 {f:2d}，乘加 ÷ 读 = {ratio:.2f}")

    print("\n第三步 Kernel 5 的配置：BM = BN = 128、BK = 8、256 个线程、每线程 8 × 8")
    print("  256 × 64 =", 256 * 64, "= 128 × 128 =", 128 * 128)
    rng = np.random.default_rng(0)
    As = rng.standard_normal((128, 8)).astype(np.float32)
    Bs = rng.standard_normal((8, 128)).astype(np.float32)
    out = block_tile(As, Bs)
    print("  按外积一步步算出的 128 × 128 和 As @ Bs 一致：", np.allclose(out, As @ Bs, atol=1e-5))
    print("  线程 (0, 0) 的小块左上角：", out[0, 0], "≈", (As @ Bs)[0, 0])

    print("\n第六步 Boehm 全景（A6000，相对 cuBLAS 23250 GFLOP/s）")
    for k, what, g in BOEHM:
        print(f"  Kernel {k:>2} {what:<14} {g:6d} GFLOP/s  {g / CUBLAS:6.1%}")


if __name__ == "__main__":
    main()
