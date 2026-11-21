"""讲义：第 8 章 · occupancy 与延迟隐藏（notes/lessons/08-occupancy.md）

运行：uv run python ch08_occupancy.py
四道门槛（线程、block 数、寄存器、shared memory）各允许几个 block，取最紧的那道。
"""

import math

# CC 10.0 每个 SM 的上限（PG 表 2）；CC 7.0、9.0 的这几项相同
LIMITS = {"threads": 2048, "blocks": 32, "regs": 65536, "smem": 233472, "max_warps": 64}


def regs_per_block(threads: int, regs_per_thread: int, round_to_256: bool = True) -> int:
    """寄存器按 warp 分配：每 warp 用 regs × 32 个，向上取整到 256 的倍数（Best Practices §11.1.1）。"""
    per_warp = regs_per_thread * 32
    if round_to_256:
        per_warp = math.ceil(per_warp / 256) * 256
    return math.ceil(threads / 32) * per_warp


def blocks_per_sm(threads: int, regs_per_thread: int = 0, smem_bytes: int = 0, round_to_256: bool = True,
                  limits: dict = LIMITS) -> dict[str, int]:
    """每道门槛各允许几个 block。"""
    big = 10**9
    r = regs_per_block(threads, regs_per_thread, round_to_256)
    return {
        "threads": limits["threads"] // threads,
        "blocks": limits["blocks"],
        "regs": limits["regs"] // r if r else big,
        "smem": limits["smem"] // smem_bytes if smem_bytes else big,
    }


def occupancy(threads: int, regs_per_thread: int = 0, smem_bytes: int = 0, round_to_256: bool = True,
              limits: dict = LIMITS) -> tuple[int, str, float]:
    """返回（每 SM 的 block 数, 卡脖子的门槛, occupancy = 驻留 warp ÷ 最多 warp）。"""
    gates = blocks_per_sm(threads, regs_per_thread, smem_bytes, round_to_256, limits)
    worst = min(gates, key=gates.get)
    nb = gates[worst]
    return nb, worst, nb * math.ceil(threads / 32) / limits["max_warps"]


def laundromat(people: int, wash: int = 40, load: int = 1) -> float:
    """洗衣房：一桶要洗 wash 分钟，放衣服 load 分钟；people 个人轮流用时机器有多少时间在干活。"""
    return min(1.0, people * load / wash)


def main():
    print("\n第一步 洗衣房：一台机器洗一桶 40 分钟，放衣服 1 分钟")
    for p in (1, 10, 40):
        print(f"  {p:2d} 个人轮流用：机器忙的比例 {laundromat(p):.1%}")

    print("\n第三步 四个例子")
    cases = [
        ("例 1 block 768 线程", {"threads": 768}),
        ("例 2 block 32 线程", {"threads": 32}),
        ("例 3 每 block 100 KB shared", {"threads": 128, "smem_bytes": 100 * 1024}),
        ("例 4 每线程 37 个寄存器", {"threads": 128, "regs_per_thread": 37}),
    ]
    for name, kw in cases:
        nb, worst, occ = occupancy(**kw)
        print(f"  {name:<22} 每 SM {nb:2d} 个 block，卡在 {worst:<7}，occupancy = {occ:.1%}")
    print("  例 4 的寄存器：37 × 32 =", 37 * 32, "→ 取整到", regs_per_block(32, 37), "个 / warp；每 block",
          regs_per_block(128, 37), "个")
    print("  100% occupancy 时每线程最多", LIMITS["regs"] // LIMITS["threads"], "个寄存器")


if __name__ == "__main__":
    main()
