"""讲义：第 1 章 · 线程层级与全局编号（notes/lessons/01-thread-hierarchy.md）

运行：uv run python ch01_thread_index.py
Mac 上没有 GPU：用 NumPy 把"每个线程算自己的 i"模拟一遍。一个 kernel 的所有线程同时跑，
在这里就是一次算出所有线程的编号表（grid × block）。
"""

import numpy as np


def global_index(block_idx: int, block_dim: int, thread_idx: int) -> int:
    """全局编号：i = blockIdx.x × blockDim.x + threadIdx.x。"""
    return block_idx * block_dim + thread_idx


def launch_indices(grid_dim: int, block_dim: int) -> np.ndarray:
    """模拟 <<<grid_dim, block_dim>>>：返回形状 (grid_dim, block_dim) 的表，第 b 行第 t 列是那个线程的 i。"""
    block_idx = np.arange(grid_dim)[:, None]   # 每一行是一个 block
    thread_idx = np.arange(block_dim)[None, :]  # 每一列是 block 里的一个座位
    return block_idx * block_dim + thread_idx   # 广播：一次算出所有线程的 i


def ceil_div(n: int, b: int) -> int:
    """向上取整的整数除法：(n + b - 1) // b，即 C++ 里的 (n + B - 1) / B。"""
    return (n + b - 1) // b


def vec_add(a: np.ndarray, b: np.ndarray, block_dim: int, bounds_check: bool = True) -> np.ndarray:
    """模拟 vecAdd kernel：host 端向上取整开 block，kernel 端每个线程做 if (i < n) c[i] = a[i] + b[i]。"""
    n = len(a)
    c = np.zeros_like(a)
    i = launch_indices(ceil_div(n, block_dim), block_dim).ravel()  # 所有线程的 i 排成一串
    if bounds_check:
        i = i[i < n]                                                # 越界的线程"举手不答题"
    c[i] = a[i] + b[i]                                              # 不检查时这里会读写数组外面
    return c


def main():
    print("\n第二、三步 <<<4, 256>>>：每个 block 负责哪一段")
    idx = launch_indices(4, 256)
    for blk in range(4):
        print(f"  block {blk}: i = {idx[blk, 0]} … {idx[blk, -1]}")
    print("  总线程数 =", idx.size, "；不重不漏：", np.array_equal(np.sort(idx.ravel()), np.arange(1024)))

    print("\n第三步 Oxford 例子：第 1 号 block（每个 64 线程）里的第 44 号线程")
    print("  i =", global_index(1, 64, 44))

    print("\n第四步 n = 1000、block = 256：向上取整 + 越界检查")
    n, B = 1000, 256
    print(f"  n / B = {n / B:.2f}，整数除法 n // B = {n // B}，向上取整 = {ceil_div(n, B)}")
    print(f"  开了 {ceil_div(n, B) * B} 个线程，其中 {ceil_div(n, B) * B - n} 个越界，什么也不做")
    rng = np.random.default_rng(0)
    a, b = rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)
    print("  加了越界检查，结果和 a + b 一样：", np.allclose(vec_add(a, b, B), a + b))
    try:
        vec_add(a, b, B, bounds_check=False)
    except IndexError as e:
        print("  去掉越界检查：NumPy 直接报错（GPU 上不会报错，而是悄悄读写别处的内存）→", type(e).__name__)


if __name__ == "__main__":
    main()
