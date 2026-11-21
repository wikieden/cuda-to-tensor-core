"""讲义：第 4 章 · 一个完整的 CUDA 程序（notes/lessons/04-host-device-program.md）

运行：uv run python ch04_host_device.py
用一个玩具 FakeGPU 类模拟 host / device 两块内存、异步启动和"晚到"的错误；最后算一笔搬运账。
真正的 cudaMalloc / cudaMemcpy 需要 NVIDIA GPU [GPU]，这里只模拟它们的行为规则。
"""

import numpy as np


class FakeGPU:
    """一块"显存" + 一个任务队列。kernel 启动只是入队，同步时才真正执行。"""

    def __init__(self):
        self.mem: dict[str, np.ndarray] = {}   # device 内存：名字 → 数组（CPU 不能直接下标访问）
        self.queue: list = []                  # 已下单、还没做的 kernel
        self.pending_error: str | None = None  # kernel 运行中出的错，要到同步点才报告

    def malloc(self, name: str, n: int):                       # ① cudaMalloc
        self.mem[name] = np.zeros(n, dtype=np.float32)

    def memcpy_h2d(self, name: str, host: np.ndarray):         # ② H2D
        self.synchronize()
        self.mem[name][:] = host

    def launch(self, kernel, *args) -> str:                    # ③ kernel<<<...>>>：只入队，立刻返回
        self.queue.append((kernel, args))
        return "cudaSuccess"                                   # 启动参数没问题 ≠ kernel 跑成功

    def synchronize(self) -> str:                              # ④ cudaDeviceSynchronize
        for kernel, args in self.queue:
            try:
                kernel(self.mem, *args)
            except IndexError:
                self.pending_error = "cudaErrorIllegalAddress"
        self.queue.clear()
        err, self.pending_error = self.pending_error or "cudaSuccess", None
        return err

    def memcpy_d2h(self, name: str) -> tuple[str, np.ndarray]:  # ④ D2H：先等前面的 kernel 做完
        err = self.synchronize()
        return err, self.mem[name].copy()


def vec_add_kernel(mem, a: str, b: str, c: str, n: int):
    mem[c][:n] = mem[a][:n] + mem[b][:n]


def bad_kernel(mem, a: str, n: int):
    """越界写：写到 n + 5 的位置。"""
    mem[a][n + 5] = 1.0


def transfer_ms(megabytes: float, gb_per_s: float) -> float:
    """搬运时间（毫秒）= 字节数 ÷ 带宽。"""
    return megabytes * 1e6 / (gb_per_s * 1e9) * 1e3


def main():
    print("\n第二、三步 五步走 + 异步启动")
    n = 8
    gpu = FakeGPU()
    for name in ("A", "B", "C"):
        gpu.malloc(name, n)                                         # ① 分配
    gpu.memcpy_h2d("A", np.arange(n, dtype=np.float32))             # ② 拷进去
    gpu.memcpy_h2d("B", np.ones(n, dtype=np.float32))
    print("  启动返回：", gpu.launch(vec_add_kernel, "A", "B", "C", n))  # ③ 启动
    print("  刚启动完，C 里还是：", gpu.mem["C"], "（kernel 只是排进了队列）")
    err, C = gpu.memcpy_d2h("C")                                    # ④ 拷回（会先等 kernel）
    print("  拷回以后 C =", C, "，状态", err)

    print("\n第四步 错误会晚到")
    gpu2 = FakeGPU()
    gpu2.malloc("A", n)
    print("  启动越界 kernel，cudaGetLastError →", gpu2.launch(bad_kernel, "A", n))
    print("  到了同步点才冒出来 →", gpu2.synchronize())

    print("\n第五步 n = 1 亿个 float，每个数组 400 MB")
    rows = [("H2D 拷 A、B", 800, 64), ("kernel 读 A、B 写 C", 1200, 3350), ("D2H 拷 C", 400, 64)]
    total = 0.0
    for what, mb, bw in rows:
        ms = transfer_ms(mb, bw)
        total += ms
        print(f"  {what:<12} {mb:5d} MB ÷ {bw:5d} GB/s ≈ {ms:6.2f} ms")
    print(f"  合计 ≈ {total:.1f} ms，其中 kernel 只占 {transfer_ms(1200, 3350) / total:.1%}")


if __name__ == "__main__":
    main()
