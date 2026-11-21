"""讲义：第 5 章 · GPU 内存层级（notes/lessons/05-memory-hierarchy.md）

运行：uv run python ch05_memory_hierarchy.py
把 H100 每一层的容量算一遍，再用一个小模拟数"读一次、用很多次"省下多少次显存读。
延迟是量级（讲义标了〔旧知〕），代码里只用容量做计算。
"""

import numpy as np

KB, MB, GB = 2**10, 2**20, 2**30

# H100 SXM 每一层（Hopper tuning guide §4.1–4.2）
H100 = {
    "寄存器（每 SM）": 65536 * 4,          # 64K 个 32 位寄存器
    "shared memory（每 SM 最多）": 228 * KB,
    "L2（全 GPU）": 50 * MB,
    "HBM3（全 GPU）": 80 * GB,
}
SMS = 132
MAX_REGS_PER_THREAD = 255


def sm_totals() -> dict[str, float]:
    """几个讲义里的"反直觉"数字。"""
    return {
        "每 SM 寄存器 KB": H100["寄存器（每 SM）"] / KB,
        "132 个 SM 的 shared 合计 MB": SMS * H100["shared memory（每 SM 最多）"] / MB,
        "L2 MB": H100["L2（全 GPU）"] / MB,
    }


def global_reads(n_threads: int, reuse_via_shared: bool) -> int:
    """n_threads 个线程都要用同一块 8 个数的数据。

    不用 shared：每个线程各自从显存读 8 个数；用 shared：block 合作读一次（8 次），大家从 shared 里拿。
    """
    tile = 8
    return tile if reuse_via_shared else n_threads * tile


def demo_reuse(n_threads: int = 8) -> np.ndarray:
    """动画最后一帧：一块数据搬上 shared 后，被 8 个线程各用一遍（这里每个线程求一个加权和）。"""
    g = np.arange(8, dtype=np.float32)            # 显存里的数据
    s = g.copy()                                  # 读一次，放进"shared"
    w = np.arange(1, n_threads + 1)[:, None]      # 每个线程自己的权重（放寄存器）
    return (s[None, :] * w).sum(axis=1)           # 8 个线程都从 shared 里读同一块


def main():
    print("\n第二步 H100 每一层的容量")
    for k, v in H100.items():
        unit, size = next((u, s) for u, s in (("GB", GB), ("MB", MB), ("KB", KB)) if v >= s)
        print(f"  {k:<22} {v / size:>5.0f} {unit}")
    for k, v in sm_totals().items():
        print(f"  {k:<22} {v:,.1f}")
    print(f"  每线程最多 {MAX_REGS_PER_THREAD} 个寄存器")
    print("  寄存器总量 = shared + L1 的片上总量（256 KB）：", H100["寄存器（每 SM）"] == 256 * KB)

    print("\n第四步 读一次、用很多次：8 个线程都要同一块 8 个数")
    print("  各读各的：显存读", global_reads(8, False), "次；经过 shared：", global_reads(8, True), "次")
    print("  8 个线程各自算出的结果：", demo_reuse())


if __name__ == "__main__":
    main()
