"""讲义：第 9 章 · roofline、计时与 Nsight（notes/lessons/09-roofline-profiling.md）

运行：uv run python ch09_roofline.py
roofline 一条公式算上限；再在 CPU 上用 NumPy 演示"预热 + 多次取中位数 + 算有效带宽"的计时规范。
Nsight Systems / Nsight Compute 要 NVIDIA GPU [GPU]，本地只能读报告。
"""

import time

import numpy as np

H100_BF16 = {"peak": 989e12, "bw": 3.35e12}   # 稠密 BF16 Tensor Core、HBM3（规格书〔旧知〕）
A6000_FP32 = {"peak": 30e12, "bw": 768e9}     # Boehm 博客第 129 行


def ridge(peak: float, bw: float) -> float:
    """脊点 = 峰值算力 ÷ 显存带宽（FLOP / byte）。"""
    return peak / bw


def attainable(intensity: float, peak: float, bw: float) -> float:
    """可达性能 = min(峰值算力, 算术强度 × 带宽)。"""
    return min(peak, intensity * bw)


def intensity_vec_add_fp32() -> float:
    """n 次加法，读 2 个、写 1 个 float：n / 12n。"""
    return 1 / 12


def intensity_relu_bf16() -> float:
    """n 次比较，读 1 个、写 1 个 BF16：n / 4n。"""
    return 1 / 4


def intensity_gemm(n: int, bytes_per: int) -> float:
    """N × N 矩阵乘：2N³ FLOP ÷ 至少 3 个矩阵 × N² × 每元素字节。"""
    return 2 * n**3 / (3 * bytes_per * n**2)


def bound(intensity: float, peak: float, bw: float) -> str:
    return "memory-bound" if intensity < ridge(peak, bw) else "compute-bound"


def boehm_lower_bounds(n: int = 4092) -> tuple[float, float]:
    """Boehm 的估算：算要多久、搬最少的数据要多久（毫秒）。搬 = 读 A、B、C + 写 C，共 4 × N² × 4 B。"""
    flops = 2 * n**3 + n**2
    moved = 4 * n**2 * 4
    return flops / A6000_FP32["peak"] * 1e3, moved / A6000_FP32["bw"] * 1e3


def time_it(fn, warmup: int = 3, repeat: int = 20) -> float:
    """计时规范：先预热，再重复多次取中位数（秒）。GPU 上要换成 CUDA event。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def effective_bandwidth(read_bytes: int, write_bytes: int, seconds: float) -> float:
    """有效带宽 = (读字节 + 写字节) ÷ 耗时（Best Practices §9.2.2）。"""
    return (read_bytes + write_bytes) / seconds


def main():
    print("\n第二步 H100 BF16 的脊点")
    print(f"  脊点 = 989 / 3.35 ≈ {ridge(**H100_BF16):.0f} FLOP/byte")

    print("\n第二步 四个 kernel 放到 roofline 上")
    rows = [("向量加法 FP32", intensity_vec_add_fp32()), ("ReLU BF16", intensity_relu_bf16()),
            ("矩阵乘 FP32 N=4096", intensity_gemm(4096, 4)), ("矩阵乘 BF16 N=4096", intensity_gemm(4096, 2))]
    for name, ai in rows:
        print(f"  {name:<18} 强度 {ai:8.3f}  → {bound(ai, **H100_BF16)}")
    cap = attainable(intensity_vec_add_fp32(), **H100_BF16)
    print(f"  向量加法上限 = {cap / 1e12:.2f} TFLOP/s，是峰值的 {cap / H100_BF16['peak']:.3%}")

    print("\n第二步 Boehm 的 A6000（FP32 30 TFLOP/s、768 GB/s）")
    print(f"  脊点 = {ridge(**A6000_FP32):.0f}")
    t_calc, t_mem = boehm_lower_bounds()
    print(f"  4092² 矩阵乘：算 ≈ {t_calc:.2f} ms，搬 268 MB ≈ {t_mem:.2f} ms → compute-bound")

    print("\n第三步 计时规范（CPU 上的 NumPy 加法演示，数字因机器而异）")
    n = 4_000_000
    a, b = np.ones(n, np.float32), np.ones(n, np.float32)
    c = np.empty_like(a)
    t = time_it(lambda: np.add(a, b, out=c))
    print(f"  中位数耗时 {t * 1e3:.2f} ms，有效带宽 ≈ {effective_bandwidth(8 * n, 4 * n, t) / 1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
