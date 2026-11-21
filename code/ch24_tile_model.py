"""讲义：第 24 章 · Tile 模型与 kernel DSL（notes/lessons/24-tile-dsl.md）

运行：uv run python ch24_tile_model.py
同一个向量加法写两遍：SIMT（替每个线程想）和 Tile（替每个 block 想）。用 NumPy 模拟 ct.load / ct.store 的 tile space 越界处理。
cuTile / Triton 真跑要 NVIDIA GPU [GPU]。
"""

import numpy as np


def simt_vec_add(a: np.ndarray, b: np.ndarray, block_dim: int) -> np.ndarray:
    """第 1 章的写法：每个线程算 i，自己写 if (i < n)。"""
    n = len(a)
    c = np.zeros_like(a)
    grid = (n + block_dim - 1) // block_dim
    for bid in range(grid):
        for tid in range(block_dim):
            i = bid * block_dim + tid
            if i < n:
                c[i] = a[i] + b[i]
    return c


def is_valid_tile(size: int) -> bool:
    """tile 每一维必须是 2 的幂（而且编译期已知）。"""
    return size > 0 and size & (size - 1) == 0


def tile_load(arr: np.ndarray, index: int, tile: int) -> np.ndarray:
    """ct.load(arr, index=(bid,), shape=(TILE,))：取 tile space 里第 index 块；超出数组的部分补 0。"""
    out = np.zeros(tile, dtype=arr.dtype)
    part = arr[index * tile:(index + 1) * tile]
    out[:len(part)] = part
    return out


def tile_store(arr: np.ndarray, index: int, t: np.ndarray):
    """ct.store：只写回数组里存在的那部分，越界部分丢掉。"""
    seg = arr[index * t.size:(index + 1) * t.size]
    seg[:] = t[:len(seg)]


def tile_vec_add(a: np.ndarray, b: np.ndarray, tile: int) -> np.ndarray:
    """Tile 写法：没有 threadIdx、没有 if (i < n)；每个 block 处理一整块，block 里几个线程由编译器定。"""
    assert is_valid_tile(tile), "tile 大小必须是 2 的幂"
    c = np.zeros_like(a)
    for bid in range(-(-len(a) // tile)):           # ct.bid(0)
        ta = tile_load(a, bid, tile)
        tb = tile_load(b, bid, tile)
        tile_store(c, bid, ta + tb)                  # ta + tb 产生一个新 tile，ta、tb 本身不变
    return c


def main():
    print("\n第二步 n = 20：SIMT（block 8 个线程）和 Tile（TILE = 8）")
    a = np.arange(20, dtype=np.float32)
    b = np.full(20, 100, dtype=np.float32)
    print("  SIMT：", simt_vec_add(a, b, 8)[:5], "…")
    print("  Tile：", tile_vec_add(a, b, 8)[:5], "…")
    print("  两种写法一样，也等于 a + b：", np.array_equal(simt_vec_add(a, b, 8), a + b)
          and np.array_equal(tile_vec_add(a, b, 8), a + b))
    print("  最后一块 tile 的内容（越界补 0）：", tile_load(a, 2, 8))

    print("\n第一步 tile 大小必须是 2 的幂")
    for s in (8, 16, 24, 128):
        print(f"  {s:3d}: {is_valid_tile(s)}")


if __name__ == "__main__":
    main()
