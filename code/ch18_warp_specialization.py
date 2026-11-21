"""讲义：第 18 章 · Hopper ②：wgmma、warp specialization、persistent kernel（notes/lessons/18-hopper-wgmma-warp-specialization.md）

运行：uv run python ch18_warp_specialization.py
算累加器要几个寄存器、寄存器再分配够不够；再用两个 Python 生成器模拟生产者 / 消费者和 full / empty 两组 barrier。
wgmma / setmaxnreg 要 sm_90a [GPU]。
"""

CUBLAS_H100 = 716  # BF16 TFLOP/s（Pranjal 第 69 行）
PRANJAL = [(1, "CUDA core warptiling", 32), (2, "TMA + wgmma", 317), (3, "大 tile、大指令", 423),
           (4, "warp specialization", 498), (5, "2 个消费者 + 寄存器再分配", 631),
           (6, "persistent + tile 分组", 660), (7, "更快的 barrier", 704), (8, "cluster multicast", 734),
           ("9–10", "微优化、异步 store", 758), (11, "Hilbert 曲线", 764)]


def accum_regs_per_thread(m: int, n: int, threads: int = 128) -> int:
    """m × n 个 FP32 累加器分给一个 warpgroup（128 个线程），每个线程几个 32 位寄存器。"""
    return m * n // threads


def register_budget(producer_regs: int, consumer_regs: int, consumers: int = 2, threads: int = 128) -> int:
    """setmaxnreg 之后整个 block 要多少寄存器：1 个生产者 warpgroup + consumers 个消费者 warpgroup。"""
    return producer_regs * threads + consumer_regs * threads * consumers


def run_ring(n_tiles: int, qsize: int, producer_speed: int, consumer_speed: int) -> tuple[list[int], int]:
    """环形缓冲区：生产者等 empty[i] 再装 slot i，消费者等 full[i] 再用 slot i，用完 arrive empty[i]。

    两边按各自的速度（每"拍"能做几步）交替推进。返回消费者拿到的 tile 顺序，和生产者因为 slot 满而等的次数。
    """
    slots = [None] * qsize
    full = [False] * qsize           # full[i]：slot i 装满了
    empty = [True] * qsize           # 初始化：生产者"以为所有 slot 都是空的"
    produced = consumed = waits = 0
    got = []
    while consumed < n_tiles:
        for _ in range(producer_speed):
            if produced == n_tiles:
                break
            i = produced % qsize
            if not empty[i]:
                waits += 1
                break                # empty[i].wait()：这一拍没法装
            empty[i] = False
            slots[i] = produced      # TMA 搬 A_k、B_k → slot i
            full[i] = True           # TMA 完成时 full[i] 自动记账
            produced += 1
        for _ in range(consumer_speed):
            i = consumed % qsize
            if not full[i]:
                break                # full[i].wait()
            got.append(slots[i])     # wgmma 用 slot i
            full[i] = False
            empty[i] = True          # empty[i].arrive()
            consumed += 1
    return got, waits


def main():
    print("\n第一步 wgmma m64n64k16 的累加器：每个线程", accum_regs_per_thread(64, 64), "个 FP32 寄存器")

    print("\n第三步 生产者 / 消费者，QSIZE = 5，20 块")
    got, waits = run_ring(20, 5, producer_speed=2, consumer_speed=1)
    print("  消费者按顺序拿到每一块：", got == list(range(20)), "；生产者因 slot 满等了", waits, "拍")
    got, waits = run_ring(20, 5, producer_speed=1, consumer_speed=2)
    print("  生产者慢时：顺序仍对", got == list(range(20)), "，生产者从不等 empty（等了", waits, "拍）")

    print("\n第四步 128 × 256 的输出 tile 放进一个 warpgroup")
    print("  每线程", accum_regs_per_thread(128, 256), "个寄存器 > 上限 255 → 溢出")
    b = register_budget(24, 240)
    print(f"  寄存器再分配 24 / 240 / 240：240 × 128 × 2 + 24 × 128 = {b} ≤ 65536：{b <= 65536}")

    print("\n第六步 Pranjal 全景（H100 SXM、BF16，相对 cuBLAS 716 TFLOP/s）")
    for k, what, t in PRANJAL:
        print(f"  Kernel {k:>2} {what:<16} {t:4d} TFLOP/s  {t / CUBLAS_H100:5.0%}")


if __name__ == "__main__":
    main()
