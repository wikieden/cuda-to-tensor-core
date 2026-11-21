"""讲义：第 7 章 · shared memory 与 bank conflict（notes/lessons/07-shared-memory-banks.md）

运行：uv run python ch07_bank_conflict.py
第 w 个 4 字节字在 w % 32 号 bank；同一 bank 上不同地址的请求要排队，同一地址广播。
"""

import math

import numpy as np

BANKS = 32
WARP = 32


def bank(word_addr):
    """字地址（以 4 字节为单位）→ bank 号。"""
    return np.asarray(word_addr) % BANKS


def conflict_ways(word_addrs) -> int:
    """几路冲突 = 同一个 bank 上不同地址最多有几个（同一地址算广播，只算一个）。"""
    distinct = np.unique(np.asarray(word_addrs))
    return int(np.bincount(bank(distinct), minlength=BANKS).max())


def strided(stride: int) -> np.ndarray:
    """warp 里线程 t 读 s[t * stride]。"""
    return np.arange(WARP) * stride


def tile_addr(r, c, row_len: int):
    """二维 shared 数组 tile[r][c] 的字地址：r * 每行长度 + c。"""
    return np.asarray(r) * row_len + np.asarray(c)


def race_demo(with_barrier: bool, n: int = 8) -> np.ndarray:
    """第二步的例子：s[t] = g[t]；（栅栏）；y = s[n-1-t]。

    没有栅栏时模拟最坏的交错：每个线程写完自己那格马上去读，后面的线程还没写。
    """
    g = np.arange(10, 10 + n)
    s = np.full(n, -1)                    # -1 表示还没写进去的"垃圾值"
    y = np.zeros(n, dtype=int)
    if with_barrier:
        s[:] = g                          # ① 所有人先写
        y[:] = s[n - 1 - np.arange(n)]    # ③ 栅栏之后再读
    else:
        for t in range(n):                # 线程 0 先写先读，线程 1 再写再读……
            s[t] = g[t]
            y[t] = s[n - 1 - t]
    return y


def main():
    print("\n第二步 __syncthreads()：写完再读")
    print("  有栅栏：", race_demo(True))
    print("  没栅栏：", race_demo(False), "（-1 是读到了还没写的位置）")

    print("\n第四步 跨步访问 s[t * stride]")
    for s in (1, 2, 3, 4, 32):
        print(f"  stride {s:2d}: 用到 {len(np.unique(bank(strided(s)))):2d} 个 bank，"
              f"{conflict_ways(strided(s)):2d} 路冲突，gcd(stride, 32) = {math.gcd(s, 32)}")
    print("  32 个线程都读 s[0]（同一地址）：", conflict_ways(np.zeros(WARP, dtype=int)), "路——广播")

    print("\n第四、五步 二维数组按列读：warp 的线程 r = 0..31 读 tile[r][5]")
    r = np.arange(WARP)
    print("  tile[32][32]：bank 都是", np.unique(bank(tile_addr(r, 5, 32))), "→",
          conflict_ways(tile_addr(r, 5, 32)), "路冲突")
    print("  tile[32][33]：用到", len(np.unique(bank(tile_addr(r, 5, 33)))), "个 bank →",
          conflict_ways(tile_addr(r, 5, 33)), "路")
    print("  按行读 tile[5][t]：", conflict_ways(tile_addr(5, r, 32)), "路")
    print(f"  padding 多用 {32} 个 float，占 {32 / (32 * 32):.0%}")


if __name__ == "__main__":
    main()
