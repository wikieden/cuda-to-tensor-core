"""讲义：第 16 章 · 异步拷贝与多级流水线（notes/lessons/16-async-copy-pipeline.md）

运行：uv run python ch16_async_pipeline.py
用一个事件时间表模拟"搬 / 算"两条流水线和 S 份 shared buffer；再模拟带相位的异步 barrier。
cp.async 本身要 CC 8.0+ 的 GPU [GPU]。
"""


def pipeline(n: int, t_load: float, t_compute: float, stages: int) -> tuple[float, list, list]:
    """n 块数据，搬一块 t_load、算一块 t_compute，shared 里有 stages 份 buffer。

    规则：搬运一块接一块（一条显存通道）；第 i 块要等第 i − stages 块算完、buffer 空出来才能开始搬；
    计算一块接一块，第 i 块要等它搬完。stages = 1 就是"搬完再算"的同步写法。
    返回（总时间, 每块的搬运时段, 每块的计算时段）。
    """
    loads, computes = [], []
    for i in range(n):
        start = loads[-1][1] if loads else 0
        if i >= stages:
            start = max(start, computes[i - stages][1])     # 等 buffer 空
        loads.append((start, start + t_load))
        c0 = max(loads[i][1], computes[-1][1] if computes else 0)
        computes.append((c0, c0 + t_compute))
    return computes[-1][1], loads, computes


def stage_offset(stage: int, block_size: int) -> int:
    """PG §4.12.1.2 的 prefetch_kernel：第 s 份 buffer 从 s * block.size() 开始。"""
    return stage * block_size


class Barrier:
    """异步 barrier（PG §4.10）：期望 expected 次 arrive；到齐后相位翻转、计数重置。"""

    def __init__(self, expected: int):
        self.expected = expected
        self.count = expected
        self.phase = 0

    def arrive(self) -> int:
        """报到，不停下；返回 token（当前相位）。"""
        token = self.phase
        self.count -= 1
        if self.count == 0:
            self.phase ^= 1
            self.count = self.expected
        return token

    def done(self, token: int) -> bool:
        """wait(token) 能不能通过：token 那一相位已经结束（相位翻过去了）。"""
        return self.phase != token


def main():
    print("\n第一步 搬 4、算 4，8 块")
    print("  同步（搬完再算）：", pipeline(8, 4, 4, 1)[0])
    print("  双缓冲：", pipeline(8, 4, 4, 2)[0], "（≈ 4 + 8 × 4）")

    print("\n第四步 搬 4、算 2，8 块")
    for s, name in ((1, "同步"), (2, "双缓冲"), (3, "3 级")):
        print(f"  {name:<4}: {pipeline(8, 4, 2, s)[0]}")
    _, loads, _ = pipeline(8, 4, 2, 2)
    print("  双缓冲的搬运首尾相接：", loads[:3], "…")
    print("  max(搬, 算) × n =", max(4, 2) * 8, "，再加上最后一块的计算 2")

    print("\n第四步 反过来：算比搬慢（搬 2、算 4，8 块）")
    print("  同步：", pipeline(8, 2, 4, 1)[0], "；双缓冲：", pipeline(8, 2, 4, 2)[0], "≈ 算的时间 32 + 第一次搬 2")

    print("\n第四步 S 级 buffer 的起点（block.size() = 256）")
    print("  ", [stage_offset(s, 256) for s in range(3)], "；轮转：", [(k + 1) % 3 for k in range(5)])

    print("\n第五步 barrier 的相位：期望 4 次 arrive")
    bar = Barrier(4)
    tokens = [bar.arrive() for _ in range(3)]
    print("  3 个线程报到后：相位", bar.phase, "，wait 能过吗？", bar.done(tokens[0]))
    bar.arrive()
    print("  第 4 个报到后：相位", bar.phase, "，wait 能过吗？", bar.done(tokens[0]), "；计数重置为", bar.count)


if __name__ == "__main__":
    main()
