"""讲义：第 3 章 · warp 与 SIMT（notes/lessons/03-warp-simt.md）

运行：uv run python ch03_warp_divergence.py
模拟一个 warp 遇到 if / else：每条有人走的分支都要执行一遍，不走的 lane 被屏蔽。
"""

import math

import numpy as np

WARP = 32


def warp_time(take_a: np.ndarray, t_a: float, t_b: float) -> float:
    """一个 warp 执行 if (cond) a(); else b(); 的耗时。take_a 是 32 个 lane 的条件真假。

    有 lane 走 a 就执行一遍 a，有 lane 走 b 就执行一遍 b；两边都有人走就是两趟（分岔）。
    """
    return (t_a if take_a.any() else 0) + (t_b if (~take_a).any() else 0)


def block_time(block_dim: int, cond, t_a: float, t_b: float) -> list[float]:
    """整个 block 按 32 个一组切成 warp，返回每个 warp 的耗时。cond(tid) 给出每个线程的条件。"""
    tid = np.arange(block_dim)
    take_a = cond(tid)
    return [warp_time(take_a[w * WARP:(w + 1) * WARP], t_a, t_b) for w in range(math.ceil(block_dim / WARP))]


def diverged_warps(block_dim: int, cond) -> list[int]:
    """哪些 warp 内部条件不一致（分岔）。"""
    take = cond(np.arange(block_dim))
    out = []
    for w in range(math.ceil(block_dim / WARP)):
        part = take[w * WARP:(w + 1) * WARP]
        if part.any() and not part.all():
            out.append(w)
    return out


def lane_utilization(block_dim: int) -> tuple[int, int, float]:
    """block 线程数不是 32 的倍数时：（warp 数, 最后一个 warp 的线程数, 算力利用率）。"""
    warps = math.ceil(block_dim / WARP)
    last = block_dim - (warps - 1) * WARP
    return warps, last, block_dim / (warps * WARP)


def main():
    print("\n第二步 if (threadIdx.x % 2 == 0)：a、b 各耗时 10")
    t = block_time(32, lambda tid: tid % 2 == 0, 10, 10)
    print(f"  一个 warp 耗时 {t[0]}（没有分岔时只要 10），lane 效率 = {10 / t[0]:.0%}")

    print("\n第二步 改成按 warp 分组：if ((threadIdx.x / 32) % 2 == 0)")
    t = block_time(64, lambda tid: (tid // 32) % 2 == 0, 10, 10)
    print(f"  warp 0、warp 1 各耗时 {t}：每个 warp 只走一条路，没有浪费")

    print("\n第二步 常见写法，64 个线程的 block 里哪些 warp 分岔")
    print("  if (threadIdx.x % 2 == 0):", diverged_warps(64, lambda tid: tid % 2 == 0))
    print("  if (threadIdx.x < 16)    :", diverged_warps(64, lambda tid: tid < 16))
    print("  if (i < n)，n = 40       :", diverged_warps(64, lambda tid: tid < 40), "（只有最后一个 warp）")

    print("\n第三步 block 有 100 个线程")
    warps, last, util = lane_utilization(100)
    print(f"  warp 数 = {warps}，最后一个 warp 只有 {last} 个线程，空着 {WARP - last} 个 lane，利用率 = {util:.0%}")


if __name__ == "__main__":
    main()
