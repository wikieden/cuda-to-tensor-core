"""讲义：第 27 章 · 实战方法论与全书收尾（notes/lessons/27-methodology-wrap-up.md）

运行：uv run python ch27_methodology.py
把"先写对、再判断、和光速比"写成几个小工具：带容差的对拍、隐藏测试抓作弊的 kernel、按 roofline 分类、算 SOL 百分比。
"""

import numpy as np

from ch09_roofline import H100_BF16, bound, intensity_gemm, intensity_vec_add_fp32


def check(out: np.ndarray, ref: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """对拍：相对误差 + 绝对误差容差，不要求逐位相同（并行归约改变了加法顺序）。"""
    return bool(np.allclose(out, ref, rtol=rtol, atol=atol))


def honest_kernel(x: np.ndarray) -> np.ndarray:
    return np.exp(x) * 2 + 1


def cheating_kernel(x: np.ndarray) -> np.ndarray:
    """"奖励作弊"的例子：只对公开测试那个输入返回背下来的答案，其他输入随便算。"""
    if x.shape == (4,) and np.array_equal(x, np.arange(4.0)):
        return np.exp(np.arange(4.0)) * 2 + 1      # 硬编码的输出
    return np.zeros_like(x)


def run_tests(kernel, hidden: bool, seed: int = 0) -> bool:
    """公开测试只有一个固定输入；隐藏测试换 shape、换随机种子。"""
    cases = [np.arange(4.0)]
    if hidden:
        rng = np.random.default_rng(seed)
        cases += [rng.standard_normal(s) for s in (1, 7, 1000, 4097)]
    return all(check(kernel(x), np.exp(x) * 2 + 1) for x in cases)


def sol_percent(achieved: float, peak: float) -> float:
    """报"达到光速（SOL）的百分之几"，而不是"比 naive 快几倍"。"""
    return achieved / peak


def main():
    print("\n第一步 判断瓶颈：用第 9 章的数")
    for name, ai in (("向量加法 FP32", intensity_vec_add_fp32()), ("矩阵乘 BF16 N=4096", intensity_gemm(4096, 2))):
        print(f"  {name:<18} 强度 {ai:8.3f} → {bound(ai, **H100_BF16)}")

    print("\n第一步 和光速比：第 18 章 Pranjal 的 Kernel 11（764 TFLOP/s）")
    print(f"  相对 H100 稠密 BF16 峰值 989：{sol_percent(764, 989):.0%}；相对 cuBLAS 716：{764 / 716:.0%}")

    print("\n第三步 对拍要容差：float32 的两种加法顺序")
    x = np.random.default_rng(0).random(100_000).astype(np.float32)
    seq, tree = np.cumsum(x)[-1], x.sum()
    print(f"  顺序加 {seq:.4f}，分组加 {tree:.4f}；逐位相同？{seq == tree}；容差内相同？{check(seq, tree)}")

    print("\n第四步 AI 写的 kernel：隐藏测试")
    for k in (honest_kernel, cheating_kernel):
        print(f"  {k.__name__:<16} 公开测试 {run_tests(k, False)}，加上隐藏测试 {run_tests(k, True)}")
    print("  KernelBench-Verified：加速比几何平均 1.43 → 0.88，变化", f"{0.88 / 1.43 - 1:.0%}")


if __name__ == "__main__":
    main()
