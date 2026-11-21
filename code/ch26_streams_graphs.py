"""讲义：第 26 章 · streams、CUDA Graphs、PDL（notes/lessons/26-streams-graphs-pdl.md）

运行：uv run python ch26_streams_graphs.py
算启动间隙占多少；用一个小时间线模拟"同一 stream 按顺序、不同 stream 可并发、拷贝引擎和计算各一个"；
再比较普通启动、CUDA Graph、PDL 三种情况下 GPU 空了多久。真机要 NVIDIA GPU [GPU]。
"""


def overhead(n_kernels: int, run_us: float, gap_us: float) -> tuple[float, float]:
    """n 个 kernel，每个跑 run_us，每次启动间隙 gap_us：返回（总时间 ms, 间隙占比）。"""
    total = n_kernels * (run_us + gap_us)
    return total / 1000, n_kernels * gap_us / total


def schedule(tasks: list[tuple[str, str, str, float]]) -> dict[str, tuple[float, float]]:
    """tasks：(名字, stream, 引擎, 耗时)，按提交顺序给出。

    规则：同一个 stream 里严格按顺序；同一个引擎（copy / compute）一次只做一件事；其余可以重叠。
    返回每个任务的（开始, 结束）。
    """
    stream_free: dict[str, float] = {}
    busy: dict[str, list[tuple[float, float]]] = {}      # 每个引擎已经占用的时段
    out = {}
    for name, stream, engine, dur in tasks:
        start = stream_free.get(stream, 0.0)              # 同一 stream：等前一个任务做完
        for a, b in sorted(busy.get(engine, [])):         # 同一引擎：找最早一段放得下的空闲
            if start + dur <= a:
                break
            start = max(start, b)
        out[name] = (start, start + dur)
        stream_free[stream] = start + dur
        busy.setdefault(engine, []).append((start, start + dur))
    return out


def launch_timeline(n: int, run_us: float, cpu_launch_us: float, preamble_us: float, mode: str) -> float:
    """n 个依赖成链的小 kernel，每个 = 准备工作 preamble + 正式计算 run。返回总时间（微秒）。

    normal：每个 kernel 都要等 CPU 下单（cpu_launch），再做准备、再计算；
    graph ：整张图一次下单，kernel 之间不再等 CPU；
    pdl   ：在 graph 基础上，后一个 kernel 的准备工作和前一个的计算重叠（理想情况）。
    """
    if mode == "normal":
        return n * (cpu_launch_us + preamble_us + run_us)
    if mode == "graph":
        return cpu_launch_us + n * (preamble_us + run_us)
    return cpu_launch_us + preamble_us + n * run_us


def main():
    print("\n第一步 300 个 kernel、每个 8 µs、每次启动间隙 4 µs")
    total, frac = overhead(300, 8, 4)
    print(f"  总时间 ≈ {total:.1f} ms，其中间隙 {300 * 4 / 1000:.1f} ms（{frac:.0%}）")

    print("\n第二步 两个 stream：s1 拷贝后算 X，s2 算 Y（编的耗时）")
    t = schedule([("H2D(s1)", "s1", "copy", 3), ("X(s1)", "s1", "compute", 2), ("Y(s2)", "s2", "compute", 2)])
    for k, (a, b) in t.items():
        print(f"  {k:<8} {a:4.1f} → {b:4.1f}")
    print("  Y 在 s2 里，不用等 s1 的拷贝：它和 H2D 同时进行")

    print("\n第二步 拷贝与计算重叠：数据切成 2 段，每段拷 2、算 2（编的耗时）")
    one = schedule([("H2D", "s", "copy", 4), ("K", "s", "compute", 4)])
    two = schedule([("H2D a", "s1", "copy", 2), ("K a", "s1", "compute", 2),
                    ("H2D b", "s2", "copy", 2), ("K b", "s2", "compute", 2)])
    print("  一个 stream 不切分：", max(e for _, e in one.values()), "；两个 stream 切两段：", max(e for _, e in two.values()))

    print("\n第三、四步 100 个小 kernel：每个跑 8 µs、准备 2 µs、CPU 下单 4 µs（编的耗时）")
    for mode in ("normal", "graph", "pdl"):
        print(f"  {mode:<6}: {launch_timeline(100, 8, 4, 2, mode):6.0f} µs")


if __name__ == "__main__":
    main()
