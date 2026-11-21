"""讲义：第 17 章 · Hopper ①：TMA、mbarrier、cluster（notes/lessons/17-hopper-tma-mbarrier.md）

运行：uv run python ch17_tma_mbarrier.py
模拟 tensor map（按坐标搬一个 tile、越界补 0）、按字节记账的 mbarrier、cluster 里读伙伴的 shared memory。
真正的 TMA / mbarrier 要 CC 9.0 的 GPU [GPU]。
"""

import numpy as np


class TensorMap:
    """host 端写好的"仓库地图"：全局张量的形状 + tile（box）大小。这里只做二维、行优先。"""

    def __init__(self, tensor: np.ndarray, box: tuple[int, int]):
        self.tensor = tensor
        self.box = box

    def load_tile(self, x: int, y: int) -> np.ndarray:
        """cp.async.bulk.tensor.2d：把坐标 (x, y) 的 tile（第 x 块行、第 y 块列）搬出来；越界部分填 0。"""
        bh, bw = self.box
        out = np.zeros(self.box, dtype=self.tensor.dtype)
        part = self.tensor[x * bh:(x + 1) * bh, y * bw:(y + 1) * bw]   # 越界的部分 NumPy 切片自动截掉
        out[:part.shape[0], :part.shape[1]] = part
        return out


def stride_ok(n_cols: int, elem_bytes: int) -> bool:
    """TMA 要求：除最内层外的每个 stride 是 16 字节的倍数。行优先矩阵的行 stride = 列数 × 元素字节。"""
    return n_cols * elem_bytes % 16 == 0


class MBarrier:
    """shared memory 里的 64 位对象：同时记"还差几次 arrive"和"还差多少字节"。两个都清零 → 相位翻转、重置。"""

    def __init__(self, arrive_count: int):
        self.expected = arrive_count
        self.pending_arrive = arrive_count
        self.pending_tx = 0
        self.phase = 0

    def _maybe_flip(self):
        if self.pending_arrive == 0 and self.pending_tx == 0:
            self.phase ^= 1
            self.pending_arrive = self.expected

    def arrive_expect_tx(self, nbytes: int):
        """mbarrier.arrive.expect_tx：报到一次，并声明"还要来 nbytes 字节"。"""
        self.pending_tx += nbytes
        self.pending_arrive -= 1
        self._maybe_flip()

    def complete_tx(self, nbytes: int):
        """TMA 搬完一部分时硬件自动调用：扣掉相应字节。"""
        self.pending_tx -= nbytes
        self._maybe_flip()

    def try_wait(self, parity: int) -> bool:
        """mbarrier.try_wait.parity：parity 这一相位已经完成了吗。"""
        return self.phase != parity


def cluster_exchange(bufs: list[np.ndarray]) -> list[np.ndarray]:
    """cluster 大小 2：block rank r 用 map_shared_rank(buf, r ^ 1) 读伙伴的 shared memory（DSMEM）。"""
    return [bufs[r ^ 1].copy() for r in range(len(bufs))]


def main():
    print("\n第二步 tensor map：5 × 6 的矩阵，tile 4 × 4")
    A = np.arange(1, 31).reshape(5, 6)
    tm = TensorMap(A, (4, 4))
    print("  tile (0, 0)：\n", tm.load_tile(0, 0))
    print("  tile (1, 1)（大半越界，自动补 0）：\n", tm.load_tile(1, 1))
    print("  FP32 矩阵列数 8：", stride_ok(8, 4), "；列数 6：", stride_ok(6, 4), "（要先 padding）")

    print("\n第三步 mbarrier：期望 1 次 arrive，TMA 搬 16384 字节")
    bar = MBarrier(1)
    bar.arrive_expect_tx(16384)                     # ② 线程 0 报到并声明字节数
    print("  报到后、TMA 还没搬完：能过吗？", bar.try_wait(0), "，还差", bar.pending_tx, "字节")
    bar.complete_tx(16384)                          # ③ TMA 搬完，硬件自动记账
    print("  TMA 搬完：能过吗？", bar.try_wait(0), "，相位翻成", bar.phase)
    bar.arrive_expect_tx(16384)
    print("  下一圈用 parity 1 等：", bar.try_wait(1), "（这一圈的数据还没到）")

    print("\n第四步 cluster 里两个 block 互读 shared memory")
    bufs = [np.full(4, 10.0), np.full(4, 20.0)]
    print("  block 0 读到伙伴的：", cluster_exchange(bufs)[0], "；block 1 读到：", cluster_exchange(bufs)[1])


if __name__ == "__main__":
    main()
