"""讲义：第 25 章 · 编译与部署（notes/lessons/25-compile-deploy.md）

运行：uv run python ch25_compile_deploy.py
把 PG §1.3.4 的兼容规则写成函数：一份 cubin / PTX 能不能在某张卡上跑；fatbin 里装了好几份时驱动挑哪一份。
真正编译要 nvcc / ptxas [GPU 工具链]；Mac 上可以用 Godbolt 的在线 nvcc 看 PTX / SASS。
"""


def parse(target: str) -> tuple[str, int, int, bool]:
    """'sm_86' → ('sm', 8, 6, False)；'compute_90a' → ('compute', 9, 0, True)。"""
    kind, ver = target.split("_")
    arch_specific = ver.endswith("a")
    ver = ver.rstrip("a")
    return kind, int(ver[:-1]), int(ver[-1]), arch_specific


def can_run(target: str, cc: tuple[int, int]) -> str | None:
    """返回 'cubin'（直接跑）、'JIT'（驱动把 PTX 现场编译）或 None（跑不了）。"""
    kind, major, minor, a = parse(target)
    if a:                                              # sm_90a / sm_100a：只认同一个架构
        return ("cubin" if kind == "sm" else "JIT") if (major, minor) == cc else None
    if kind == "sm":                                   # cubin：同一大版本、小版本不低于它
        return "cubin" if cc[0] == major and cc[1] >= minor else None
    return "JIT" if cc >= (major, minor) else None     # PTX：任何更新的卡都能 JIT


def pick_from_fatbin(targets: list[str], cc: tuple[int, int]) -> tuple[str, str] | None:
    """驱动从 fatbin 里挑：能直接跑的 cubin 优先（选版本最高的），没有再用 PTX JIT。"""
    ok = [(t, can_run(t, cc)) for t in targets]
    ok = [(t, how) for t, how in ok if how]
    if not ok:
        return None
    ok.sort(key=lambda th: (th[1] == "cubin", parse(th[0])[1:3]), reverse=True)
    return ok[0]


def nvcc_flags(pairs: list[tuple[str, str]]) -> str:
    """-gencode arch=compute_XX,code=sm_XX 的命令行。"""
    return " ".join(f"-gencode arch={a},code={c}" for a, c in pairs)


def main():
    print("\n第三步 规则 1：cubin 只往同一大版本的更高小版本兼容（以 sm_80 为例）")
    for cc in ((8, 0), (8, 6), (8, 9), (9, 0)):
        print(f"  sm_80 cubin 在 CC {cc[0]}.{cc[1]} 上：{can_run('sm_80', cc)}")
    print("\n第三步 规则 2：PTX 能向前兼容（JIT）")
    for cc in ((7, 5), (8, 0), (9, 0), (10, 0)):
        print(f"  compute_80 PTX 在 CC {cc[0]}.{cc[1]} 上：{can_run('compute_80', cc)}")
    print("\n第三步 规则 3：带 a 的只认自己")
    for cc in ((9, 0), (10, 0)):
        print(f"  sm_90a 在 CC {cc[0]}.{cc[1]} 上：{can_run('sm_90a', cc)}")

    print("\n第二步 讲义里那条 nvcc 命令装进 fatbin 的三份")
    pairs = [("compute_80", "sm_80"), ("compute_90a", "sm_90a"), ("compute_90", "compute_90")]
    print("  nvcc", nvcc_flags(pairs), "my.cu")
    fatbin = [c for _, c in pairs]
    for cc in ((8, 0), (8, 6), (9, 0), (10, 0), (7, 5)):
        print(f"  CC {cc[0]}.{cc[1]:<2} 的卡挑：{pick_from_fatbin(fatbin, cc)}")


if __name__ == "__main__":
    main()
