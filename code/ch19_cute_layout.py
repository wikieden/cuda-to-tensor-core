"""讲义：第 19 章 · CuTe layout（notes/lessons/19-cute-layout.md）

运行：uv run python ch19_cute_layout.py
用 Python 写一个最小的 Layout = Shape : Stride（支持嵌套），再演示 colexicographic 顺序、按 tile 切块和 swizzle。
真正的 CuTe 是 C++ 模板 / CuTe DSL，这里只模拟它的下标算法。
"""

import numpy as np


def _flatten(t):
    """把嵌套的 tuple 摊平成一串整数。"""
    if isinstance(t, int):
        return [t]
    return [x for e in t for x in _flatten(e)]


def left_strides(shape) -> tuple:
    """不写 Stride 时的默认 LayoutLeft：步长是 Shape 从左往右的累乘（广义列优先）。"""
    out, acc = [], 1
    for s in shape:
        out.append(acc)
        acc *= s
    return tuple(out)


class Layout:
    """Layout = (Shape, Stride)：坐标 → 下标 = 坐标逐项乘步长再相加。Shape / Stride 可以嵌套。"""

    def __init__(self, shape, stride=None):
        self.shape = shape
        self.stride = stride if stride is not None else left_strides(shape)

    def size(self) -> int:
        return int(np.prod(_flatten(self.shape)))

    def __call__(self, coord):
        """coord 可以是和 Shape 同样嵌套的坐标，也可以是一个一维整数（按 colexicographic 顺序换成多维坐标）。"""
        if isinstance(coord, int):
            coord = idx2crd(coord, self.shape)
        return sum(c * d for c, d in zip(_flatten(coord), _flatten(self.stride)))

    def cosize(self) -> int:
        """用到的最大下标 + 1（存储要开多大）。"""
        return max(self(i) for i in range(self.size())) + 1

    def __repr__(self):
        return f"{self.shape}:{self.stride}"


def idx2crd(i: int, shape):
    """colexicographic 顺序：最左边的维变化最快。对嵌套 Shape 递归拆。"""
    if isinstance(shape, int):
        return i
    out = []
    for s in shape:
        n = int(np.prod(_flatten(s)))
        out.append(idx2crd(i % n, s))
        i //= n
    return tuple(out)


def zipped_divide_2d(rows: int, cols: int, tile: tuple[int, int]) -> np.ndarray:
    """8 × 8 列优先矩阵按 4 × 4 分块：结果 [块内坐标 (i, j), 块号 (bx, by)] → 原下标，形状 ((4, 4), (2, 2))。"""
    L = Layout((rows, cols))                      # (8, 8):(1, 8)
    th, tw = tile
    out = np.zeros((th, tw, rows // th, cols // tw), dtype=int)
    for bx in range(rows // th):
        for by in range(cols // tw):
            for i in range(th):
                for j in range(tw):
                    out[i, j, bx, by] = L((bx * th + i, by * tw + j))
    return out


def swizzle(x: int, b: int = 3, m: int = 3, s: int = 3) -> int:
    """CuTe 的 Swizzle<B, M, S>：取下标第 [M+S, M+S+B) 位，异或到第 [M, M+B) 位上。"""
    mask = ((1 << b) - 1) << (m + s)
    return x ^ ((x & mask) >> s)


def main():
    print("\n第一步 同一个 4 × 2，三种 Stride")
    coords = [(0, 0), (1, 0), (0, 1), (3, 1)]
    for L in (Layout((4, 2), (1, 4)), Layout((4, 2), (2, 1)), Layout((4, 2), (1, 5))):
        print(f"  {L!s:<14}", {c: L(c) for c in coords})
    print("  不写 Stride：", Layout((4, 2)), "（默认左边快，和 C 数组的行优先相反）")

    print("\n第二步 colexicographic：Shape (4, 2) 的一维坐标 0..7")
    print("  ", [idx2crd(i, (4, 2)) for i in range(8)])
    print("  嵌套 (4, (2, 2)):(2, (1, 8))：坐标 (1, (1, 1)) →", Layout((4, (2, 2)), (2, (1, 8)))((1, (1, 1))))

    print("\n第三步 8 × 8 按 4 × 4 分块：块 (1, 0) 的块内下标")
    d = zipped_divide_2d(8, 8, (4, 4))
    print("  结果形状 ((4, 4), (2, 2)) =", d.shape)
    print(d[:, :, 1, 0])
    full = np.arange(64).reshape(8, 8, order="F")   # 列优先存放的 8 × 8，元素值就是它的下标
    print("  和直接切片 full[4:8, 0:4] 一样：", np.array_equal(d[:, :, 1, 0], full[4:8, 0:4]))

    print("\n第五步 Swizzle<3, 3, 3>：0..511 映射后仍是 0..511 的一个排列")
    sw = [swizzle(i) for i in range(512)]
    print("  前 10 个：", sw[:10], "；第 64..73 个：", sw[64:74])
    print("  一一对应：", sorted(sw) == list(range(512)))


if __name__ == "__main__":
    main()
