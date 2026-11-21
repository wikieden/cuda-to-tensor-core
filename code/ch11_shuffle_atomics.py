"""讲义：第 11 章 · warp shuffle 与原子操作（notes/lessons/11-warp-shuffle-atomics.md）

运行：uv run python ch11_shuffle_atomics.py
用 NumPy 数组当作"一个 warp 每个 lane 的寄存器"，模拟 __shfl_down_sync / __shfl_xor_sync；
再用随机交错的读-改-写模拟"普通写会丢账、原子加不会"。
"""

import numpy as np


def shfl_down(v: np.ndarray, d: int) -> np.ndarray:
    """lane t 拿 lane t + d 的值；超出 warp 的 lane 拿回自己的值（CUDA 的规定）。"""
    width = len(v)
    src = np.arange(width) + d
    src[src >= width] = np.arange(width)[src >= width]
    return v[src]


def shfl_up(v: np.ndarray, d: int) -> np.ndarray:
    """lane t 拿 lane t − d 的值；t < d 的拿回自己的值。"""
    width = len(v)
    src = np.arange(width) - d
    src[src < 0] = np.arange(width)[src < 0]
    return v[src]


def shfl_xor(v: np.ndarray, m: int) -> np.ndarray:
    """lane t 拿 lane (t XOR m) 的值。"""
    return v[np.arange(len(v)) ^ m]


def warp_sum_down(v: np.ndarray) -> list[np.ndarray]:
    """for d = width/2 … 1：v += shfl_down(v, d)。返回每一步之后所有 lane 的值。"""
    v = v.copy()
    steps = []
    d = len(v) // 2
    while d > 0:
        v = v + shfl_down(v, d)
        steps.append(v.copy())
        d //= 2
    return steps


def warp_sum_xor(v: np.ndarray) -> np.ndarray:
    """蝴蝶归约：for m = 1, 2, 4…：v += shfl_xor(v, m)；最后每个 lane 都是总和。"""
    v = v.copy()
    m = 1
    while m < len(v):
        v = v + shfl_xor(v, m)
        m *= 2
    return v


def counter(n_threads: int, atomic: bool, seed: int = 0) -> int:
    """n_threads 个线程各做一次 total += 1。

    普通写：每个线程分"读旧值"和"写旧值 + 1"两步，各线程的这两步被随机交错 → 有的加法被覆盖。
    原子加：读-改-写不可打断，一个一个排队。
    """
    total = 0
    if atomic:
        for _ in range(n_threads):
            total += 1
        return total
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.repeat(np.arange(n_threads), 2))  # 每个线程号出现两次：第一次是读，第二次是写
    seen: dict[int, int] = {}
    for t in order:
        if t not in seen:
            seen[t] = total          # 读旧值（放进寄存器）
        else:
            total = seen[t] + 1      # 写回旧值 + 1：期间别人写的都被覆盖
    return total


def main():
    v = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print("\n第二步 __shfl_down_sync，8 个 lane，d = 4, 2, 1")
    for d, row in zip((4, 2, 1), warp_sum_down(v)):
        print(f"  d = {d}: lane 0–{d - 1} = {row[:d]}（其余 lane 也在算，但没人用）")
    print("  lane 0 =", warp_sum_down(v)[-1][0])

    print("\n第二步 __shfl_xor_sync：每个 lane 都拿到总和")
    print("  ", warp_sum_xor(v))

    print("\n第二步 32 个 lane：5 步")
    rng = np.random.default_rng(1)
    w = rng.integers(0, 10, 32)
    print(f"  步数 = {len(warp_sum_down(w))}，lane 0 = {warp_sum_down(w)[-1][0]}，np.sum = {w.sum()}")

    print("\n第四步 1000 个线程各加 1")
    print("  atomicAdd：", counter(1000, atomic=True))
    print("  total += 1（随机交错）：", counter(1000, atomic=False), "——丢了很多次加法，每次跑结果都可能不同")


if __name__ == "__main__":
    main()
