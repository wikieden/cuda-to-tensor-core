"""讲义：第 12 章 · 前缀和 scan（notes/lessons/12-scan.md）

运行：uv run python ch12_scan.py
Hillis–Steele、Blelloch 一轮一轮模拟出来，数加法次数；再用 shuffle 做 warp 内 scan、三段式做跨 block scan。
"""

import numpy as np

from ch11_shuffle_atomics import shfl_up


def inclusive(v) -> np.ndarray:
    """v_j = 前 j 个（含自己）之和。"""
    return np.cumsum(v)


def exclusive(v) -> np.ndarray:
    """v_j = 前面（不含自己）之和，v_0 = 0。"""
    return np.concatenate([[0], np.cumsum(v)[:-1]])


def hillis_steele(v) -> tuple[list[list[int]], int]:
    """每轮 temp[t] += temp[t − d]（t ≥ d），d = 1, 2, 4…。先全员读完再写（两道栅栏）。返回每轮结果和加法次数。"""
    temp = list(v)
    rows, adds, d = [], 0, 1
    while d < len(temp):
        old = temp[:]                                   # 第一道栅栏之后：所有人先读旧值
        temp = [old[t] + (old[t - d] if t >= d else 0) for t in range(len(old))]
        adds += len(old) - d                            # 第二道栅栏之后才写
        rows.append(temp[:])
        d *= 2
    return rows, adds


def blelloch(v) -> tuple[list[int], int]:
    """上扫（保留中间和的树形归约）+ 下扫（根置 0，左传自己、右传自己 + 左），得到 exclusive scan。"""
    temp, n, adds = list(v), len(v), 0
    d = 1
    while d < n:                                         # 上扫
        for t in range(2 * d - 1, n, 2 * d):
            temp[t] += temp[t - d]
            adds += 1
        d *= 2
    temp[-1] = 0
    d = n // 2
    while d >= 1:                                        # 下扫
        for t in range(2 * d - 1, n, 2 * d):
            temp[t - d], temp[t] = temp[t], temp[t] + temp[t - d]
            adds += 1
        d //= 2
    return temp, adds


def warp_scan(v: np.ndarray) -> np.ndarray:
    """warp 内 inclusive scan：for d = 1, 2, 4…：t = shfl_up(v, d)；lane ≥ d 的 v += t。"""
    v = v.copy()
    lane = np.arange(len(v))
    d = 1
    while d < len(v):
        t = shfl_up(v, d)
        v = np.where(lane >= d, v + t, v)
        d *= 2
    return v


def three_phase_scan(v: np.ndarray, block: int) -> np.ndarray:
    """跨 block：① 每个 block 局部 scan 并写出总和；② 对 block 总和做 exclusive scan；③ 每个 block 加回前面的总和。"""
    parts = v.reshape(-1, block)
    local = np.cumsum(parts, axis=1)              # kernel ①
    offsets = exclusive(local[:, -1])             # kernel ②
    return (local + offsets[:, None]).ravel()     # kernel ③


def main():
    v = [3, 1, 4, 1, 5, 9, 2, 6]
    print("\n第一步 inclusive / exclusive")
    print("  inclusive:", inclusive(v))
    print("  exclusive:", exclusive(v))

    print("\n第二步 Hillis–Steele")
    rows, adds = hillis_steele(v)
    for d, row in zip((1, 2, 4), rows):
        print(f"  d = {d}: {row}")
    print("  加法次数 =", adds, "（顺序算法只要", len(v) - 1, "次）")

    print("\n第三步 Blelloch")
    ex, adds = blelloch(v)
    print("  结果（exclusive）:", ex, "；加法次数 =", adds)

    print("\n第四步 warp 内用 __shfl_up_sync（8 个 lane）")
    print("  ", warp_scan(np.array(v)))

    print("\n第五步 三段式：32 个数分成 4 个 block，每个 8 个")
    rng = np.random.default_rng(0)
    big = rng.integers(0, 10, 32)
    print("  和 np.cumsum 一样：", np.array_equal(three_phase_scan(big, 8), np.cumsum(big)))


if __name__ == "__main__":
    main()
