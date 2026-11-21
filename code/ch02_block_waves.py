"""讲义：第 2 章 · block 调度与波（notes/lessons/02-block-scheduling.md）

运行：uv run python ch02_block_waves.py
用 Python 模拟 block scheduler：谁空派给谁、顺序不保证；再算波数和尾部浪费。
"""

import math

import numpy as np


def schedule(grid: int, sms: int, per_sm: int, seed: int) -> list[list[int]]:
    """模拟调度器：每一波把还没做的 block 派到空槽位上；同一波里谁先做完是随机的（seed 决定）。

    返回每一波实际派出去的 block 编号。
    """
    rng = np.random.default_rng(seed)
    todo = list(range(grid))
    waves = []
    while todo:
        slots = sms * per_sm
        batch, todo = todo[:slots], todo[slots:]
        waves.append([int(x) for x in rng.permutation(batch)])  # 完成顺序被打乱
    return waves


def run_blocks(data: np.ndarray, block_dim: int, order: list[int]) -> np.ndarray:
    """每个 block 只写自己那一段的部分和（不读别的 block 的结果），按给定顺序执行。"""
    out = np.zeros(math.ceil(len(data) / block_dim))
    for b in order:
        out[b] = data[b * block_dim:(b + 1) * block_dim].sum()
    return out


def wave_stats(grid: int, sms: int, per_sm: int) -> tuple[int, int, float]:
    """波数 = ⌈grid / (SM 数 × 每 SM 的 block 数)⌉；返回（波数, 最后一波的 block 数, 利用率）。"""
    per_wave = sms * per_sm
    waves = math.ceil(grid / per_wave)
    last = grid - (waves - 1) * per_wave
    return waves, last, grid / (waves * per_wave)


def warp_of(thread_idx: int) -> int:
    """block 里第 thread_idx 号线程属于第几个 warp（每 32 个一组，切法固定）。"""
    return thread_idx // 32


def main():
    print("\n第二步 block 之间不互相等：换三种完成顺序，结果都一样")
    data = np.arange(80, dtype=np.float64)
    results = []
    for seed in range(3):
        order = [b for wave in schedule(10, 2, 2, seed) for b in wave]
        results.append(run_blocks(data, 8, order))
        print(f"  seed {seed} 的执行顺序 {order} → 部分和 {results[-1][:3]}…")
    print("  三次结果相同：", all(np.array_equal(results[0], r) for r in results))

    print("\n第三步 4 个 SM、每个放 2 个 block、grid = 20")
    print("  每一波派出：", [len(w) for w in schedule(20, 4, 2, 0)])
    waves, last, util = wave_stats(20, 4, 2)
    print(f"  波数 = {waves}，最后一波 {last} 个，利用率 = {util:.0%}")

    print("\n第三步 H100：132 个 SM，每个放 1 个 block")
    for grid in (132, 133, 264):
        waves, last, util = wave_stats(grid, 132, 1)
        print(f"  grid {grid:3d}: 波数 {waves}，最后一波 {last:3d} / 132，利用率 {util:.1%}")
    print("  grid 只有 50 个时，从头闲到尾的 SM 数 =", 132 - 50)

    print("\n第四步 block 切成 warp：线程号 → warp 号")
    for t in (0, 31, 32, 63):
        print(f"  线程 {t:2d} → warp {warp_of(t)}")


if __name__ == "__main__":
    main()
