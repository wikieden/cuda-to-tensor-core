# CUDA 从线程到 Tensor Core

**在线阅读：<https://wikieden.github.io/cuda-to-tensor-core/>** · [讲义目录](https://wikieden.github.io/cuda-to-tensor-core/lessons/index.html) · [全部动画](https://wikieden.github.io/cuda-to-tensor-core/animations/index.html) · [复习卡片](https://wikieden.github.io/cuda-to-tensor-core/review.html)

> 27 章 · 27 个动画（GIF + 可交互页面）· 27 本 notebook · 64 张复习卡片 · 覆盖 study-plan 阶段 1–12（含 9.5）的理论部分；每章的数字都用 `solutions/chapter-code/` 的代码算过、测试核对过

## 这本书讲什么

从"一个 for 循环怎么交给成千上万个线程"讲起，一步步走到 2026 年 AI kernel 的前沿：线程层级、warp、内存层级、合并访存、bank conflict、occupancy、roofline；归约、shuffle、scan；从 naive 到 94% cuBLAS 的 SGEMM；Tensor Core 四代指令；Ampere 异步拷贝、Hopper 的 TMA / wgmma / warp specialization、CuTe、Blackwell 的 TMEM 与块缩放低精度；online softmax、FlashAttention 1–4、推理 kernel；最后是 Tile 模型与各种 DSL、编译部署、CUDA Graphs / PDL，以及一套优化方法论。

- **适合谁**：会写 C / C++ 或 Python、想看懂并亲手写 GPU kernel 的人。不需要事先懂硬件；要用到的数学只有向上取整、最大公约数、指数函数。
- **读完能做什么**：读懂 Programming Guide、Boehm / Pranjal 的 GEMM worklog、Colfax 的 CUTLASS 教程、FlashAttention 2/3/4 论文；按 study-plan 的里程碑 M0–M11 动手写 kernel；拿到一个慢 kernel 知道先查什么。
- **怎么读**：按章顺序读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。第 1–12 章在 Mac 上就能配合 cubecl-lab 练；第 13 章以后的动手部分需要云 GPU（study-plan 标了最低卡型）。

## 讲解的特点

- **一次一小步**：每个概念按"白话类比 → 代码 / 公式 → 小数字亲手算一遍"讲，算例全部由 `chapter-code` 的测试核对。
- **几条主线贯穿全书**：考场与教室（线程编号）、32 个窗口的食堂（bank）、洗衣房（延迟隐藏）、Boehm 的 A6000 SGEMM 表（第 13–14 章）、Pranjal 的 H100 GEMM 表（第 18 章）、online softmax 的 (m, ℓ)（第 21–23 章反复出现）。
- **每章一个动画**：关键原理做成可交互页面，▶ 播放、→ 单步、↺ 从头开始，开关切换不同情形；GIF 版直接嵌在正文里。
- **对照原文**：讲到 Programming Guide、Best Practices、PTX ISA、论文、博客的地方都标了本地文本的章节号和行号（如 PG §2.3.4.1，`text/cuda-programming-guide.txt` 第 3743 行），没有本地文本的标〔旧知〕。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释，全书缩写见 [缩写与术语表](https://wikieden.github.io/cuda-to-tensor-core/lessons/appendix-glossary.html)。
- **每章都有代码对照**：Mac 上没有 NVIDIA GPU，所以每章附录用 NumPy 在 CPU 上**模拟**同样的规则（线程编号、32 字节一箱、32 个 bank、片段布局、流水线时间线、mbarrier 记账……），把讲义里的数字算一遍，配测试核对、配 notebook 一格一格跑；需要真 GPU 才能看到的部分标 `[GPU]`。只算讲过的例子，不提前给练习答案。
- **每章收尾有口诀**，全部口诀汇总在本页末尾。

## 示例动画

<table>
<tr>
<td width="33%" valign="top"><img src="https://wikieden.github.io/cuda-to-tensor-core/animations/07-bank-conflict.gif" width="100%" alt="第 7 章 · bank conflict"><br><b>第 7 章 · bank conflict</b><br>按列读 32 路冲突，每行多放一个就散开<br><a href="https://wikieden.github.io/cuda-to-tensor-core/lessons/07-shared-memory-banks.html">讲义</a> · <a href="https://wikieden.github.io/cuda-to-tensor-core/animations/07-bank-conflict.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/cuda-to-tensor-core/animations/18-warp-specialization.gif" width="100%" alt="第 18 章 · warp specialization"><br><b>第 18 章 · warp specialization</b><br>生产者搬、消费者算，full / empty 两盏灯<br><a href="https://wikieden.github.io/cuda-to-tensor-core/lessons/18-hopper-wgmma-warp-specialization.html">讲义</a> · <a href="https://wikieden.github.io/cuda-to-tensor-core/animations/18-warp-specialization.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/cuda-to-tensor-core/animations/22-flashattention.gif" width="100%" alt="第 22 章 · FlashAttention"><br><b>第 22 章 · FlashAttention</b><br>一个 Q 块扫过 K、V，S 用完即丢<br><a href="https://wikieden.github.io/cuda-to-tensor-core/lessons/22-flashattention.html">讲义</a> · <a href="https://wikieden.github.io/cuda-to-tensor-core/animations/22-flashattention.html">▶ 交互</a></td>
</tr>
<tr>
<td width="33%" valign="top"><img src="https://wikieden.github.io/cuda-to-tensor-core/animations/06-coalescing.gif" width="100%" alt="第 6 章 · 合并访存"><br><b>第 6 章 · 合并访存</b><br>32 个地址落在几箱：相邻读 4 箱，跨步读一人一箱<br><a href="https://wikieden.github.io/cuda-to-tensor-core/lessons/06-coalescing.html">讲义</a> · <a href="https://wikieden.github.io/cuda-to-tensor-core/animations/06-coalescing.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/cuda-to-tensor-core/animations/14-register-tiling.gif" width="100%" alt="第 14 章 · 寄存器分块"><br><b>第 14 章 · 寄存器分块</b><br>读一列 A、一行 B，外积出一个小方块<br><a href="https://wikieden.github.io/cuda-to-tensor-core/lessons/14-sgemm-register-tiling.html">讲义</a> · <a href="https://wikieden.github.io/cuda-to-tensor-core/animations/14-register-tiling.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/cuda-to-tensor-core/animations/21-online-softmax.gif" width="100%" alt="第 21 章 · online softmax"><br><b>第 21 章 · online softmax</b><br>边读边改账：遇到更大的数，旧账乘 e 的差<br><a href="https://wikieden.github.io/cuda-to-tensor-core/lessons/21-online-softmax-fusion.html">讲义</a> · <a href="https://wikieden.github.io/cuda-to-tensor-core/animations/21-online-softmax.html">▶ 交互</a></td>
</tr>
</table>

全部动画见 [动画目录](https://wikieden.github.io/cuda-to-tensor-core/animations/index.html)。

## 目录

每章都配一个动画：**GIF** 在讲义正文里直接播放，**▶ 交互**是可以点按钮的页面（本地看需要起服务，见 [动画目录](https://wikieden.github.io/cuda-to-tensor-core/animations/index.html)"跑起来看"）。

| 章 | 阶段 | 主题 | 一句话精要 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|---|
| [1](https://wikieden.github.io/cuda-to-tensor-core/lessons/01-thread-hierarchy.html) | 1 | 线程层级与全局编号 | 只写循环体，一人领一个；向上取整 + 越界检查 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/01-thread-index.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/01-thread-index.html) | C01–C03 | 已讲，练习待作答 |
| [2](https://wikieden.github.io/cuda-to-tensor-core/lessons/02-block-scheduling.html) | 1 | block 调度与波 | block 之间不许等；最后一波不满是尾巴 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/02-block-waves.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/02-block-waves.html) | C04–C05 | 正文完成，练习待做 |
| [3](https://wikieden.github.io/cuda-to-tensor-core/lessons/03-warp-simt.html) | 1 | warp 与 SIMT | 32 人一个口令；分岔只在 warp 里 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/03-warp-divergence.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/03-warp-divergence.html) | C06–C08 | 正文完成，练习待做 |
| [4](https://wikieden.github.io/cuda-to-tensor-core/lessons/04-host-device-program.html) | 1 | 完整程序：分配、拷贝、启动、同步、查错 | 启动是下单不是等饭；错误会晚到 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/04-host-device-timeline.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/04-host-device-timeline.html) | C09–C10 | 正文完成，练习待做 |
| [5](https://wikieden.github.io/cuda-to-tensor-core/lessons/05-memory-hierarchy.html) | 2 | 内存层级 | 越近越快越小；local 不在本地 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/05-memory-hierarchy.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/05-memory-hierarchy.html) | C11–C12 | 正文完成，练习待做 |
| [6](https://wikieden.github.io/cuda-to-tensor-core/lessons/06-coalescing.html) | 2 | 合并访存 | 32 字节一箱；跨步 s 利用率 1/s | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/06-coalescing.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/06-coalescing.html) | C13–C14 | 正文完成，练习待做 |
| [7](https://wikieden.github.io/cuda-to-tensor-core/lessons/07-shared-memory-banks.html) | 2 | shared memory 与 bank conflict | 冲突路数 = gcd(跨步, 32)；[32][33] 散开 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/07-bank-conflict.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/07-bank-conflict.html) | C15–C17 | 正文完成，练习待做 |
| [8](https://wikieden.github.io/cuda-to-tensor-core/lessons/08-occupancy.html) | 2 | occupancy 与延迟隐藏 | 四道门槛看最紧；不是越高越好 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/08-occupancy.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/08-occupancy.html) | C18–C19 | 正文完成，练习待做 |
| [9](https://wikieden.github.io/cuda-to-tensor-core/lessons/09-roofline-profiling.html) | 2 | roofline、计时、Nsight | 低于脊点看带宽，高于脊点看算力 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/09-roofline.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/09-roofline.html) | C20–C22 | 正文完成，练习待做 |
| [10](https://wikieden.github.io/cuda-to-tensor-core/lessons/10-reduction.html) | 3 | 归约 | 前一半加后一半；先 grid-stride 再淘汰赛 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/10-reduction-tree.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/10-reduction-tree.html) | C23–C24 | 正文完成，练习待做 |
| [11](https://wikieden.github.io/cuda-to-tensor-core/lessons/11-warp-shuffle-atomics.html) | 3 | warp shuffle 与原子操作 | 32 个数 5 步加完；原子不丢账但排队 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/11-warp-shuffle.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/11-warp-shuffle.html) | C25–C27 | 正文完成，练习待做 |
| [12](https://wikieden.github.io/cuda-to-tensor-core/lessons/12-scan.html) | 3 | 前缀和 scan | Hillis–Steele 看远一倍；Blelloch 上树下树 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/12-scan.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/12-scan.html) | C28–C29 | 正文完成，练习待做 |
| [13](https://wikieden.github.io/cuda-to-tensor-core/lessons/13-sgemm-tiling.html) | 4 | SGEMM ①：合并与 shared 分块 | 一个数被读几千次，搬进 shared 用 32 次 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/13-gemm-tiling.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/13-gemm-tiling.html) | C30–C31 | 正文完成，练习待做 |
| [14](https://wikieden.github.io/cuda-to-tensor-core/lessons/14-sgemm-register-tiling.html) | 4 | SGEMM ②：寄存器分块、向量化、warptiling | 读的是边、算的是面积；13% → 94% | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/14-register-tiling.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/14-register-tiling.html) | C32–C33 | 正文完成，练习待做 |
| [15](https://wikieden.github.io/cuda-to-tensor-core/lessons/15-tensor-core-mma.html) | 5 | Tensor Core：WMMA、mma.sync、ldmatrix | 一个 warp 一条指令做一个小矩阵乘 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/15-mma-fragment.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/15-mma-fragment.html) | C34–C36 | 正文完成，练习待做 |
| [16](https://wikieden.github.io/cuda-to-tensor-core/lessons/16-async-copy-pipeline.html) | 6 | 异步拷贝与多级流水线 | 搬这块时算上块；慢的那步决定总时间 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/16-async-pipeline.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/16-async-pipeline.html) | C37–C38 | 正文完成，练习待做 |
| [17](https://wikieden.github.io/cuda-to-tensor-core/lessons/17-hopper-tma-mbarrier.html) | 6 | Hopper ①：TMA、mbarrier、cluster | 一个线程搬整块；按字节记账 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/17-tma-mbarrier.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/17-tma-mbarrier.html) | C39–C41 | 正文完成，练习待做 |
| [18](https://wikieden.github.io/cuda-to-tensor-core/lessons/18-hopper-wgmma-warp-specialization.html) | 6 | Hopper ②：wgmma、warp specialization、persistent | 搬运组只搬、计算组只算；常驻 132 个 block | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/18-warp-specialization.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/18-warp-specialization.html) | C42–C44 | 正文完成，练习待做 |
| [19](https://wikieden.github.io/cuda-to-tensor-core/lessons/19-cute-layout.html) | 7 | CuTe layout | Layout = Shape : Stride；换卡只换 Atom | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/19-cute-layout.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/19-cute-layout.html) | C45–C46 | 正文完成，练习待做 |
| [20](https://wikieden.github.io/cuda-to-tensor-core/lessons/20-blackwell-tcgen05.html) | 8 | Blackwell：tcgen05、TMEM、块缩放 | 累加器搬进 TMEM，一个线程发 MMA | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/20-tcgen05-tmem.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/20-tcgen05-tmem.html) | C47–C49 | 正文完成，练习待做 |
| [21](https://wikieden.github.io/cuda-to-tensor-core/lessons/21-online-softmax-fusion.html) | 9 | online softmax 与算子融合 | 边读边改账；中间结果别落地 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/21-online-softmax.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/21-online-softmax.html) | C50–C51 | 正文完成，练习待做 |
| [22](https://wikieden.github.io/cuda-to-tensor-core/lessons/22-flashattention.html) | 9 | FlashAttention 1–4 | 分块 + online softmax；每代硬件换瓶颈 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/22-flashattention.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/22-flashattention.html) | C52–C53 | 正文完成，练习待做 |
| [23](https://wikieden.github.io/cuda-to-tensor-core/lessons/23-inference-kernels.html) | 9.5 | 推理 kernel | decode 强度 ≈ batch；分页、分段、合并 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/23-paged-kv.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/23-paged-kv.html) | C54–C55 | 正文完成，练习待做 |
| [24](https://wikieden.github.io/cuda-to-tensor-core/lessons/24-tile-dsl.html) | 10 | Tile 模型与 DSL | 替 block 想而不是替线程想；先 Triton 写对 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/24-simt-vs-tile.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/24-simt-vs-tile.html) | C56–C57 | 正文完成，练习待做 |
| [25](https://wikieden.github.io/cuda-to-tensor-core/lessons/25-compile-deploy.html) | 11 | 编译与部署 | PTX 给未来、cubin 给现在；C++ kernel + Rust host | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/25-compile-pipeline.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/25-compile-pipeline.html) | C58–C59 | 正文完成，练习待做 |
| [26](https://wikieden.github.io/cuda-to-tensor-core/lessons/26-streams-graphs-pdl.html) | 12 | streams、CUDA Graphs、PDL | Graph 省 CPU 下单，PDL 省 GPU 空隙 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/26-launch-timeline.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/26-launch-timeline.html) | C60–C62 | 正文完成，练习待做 |
| [27](https://wikieden.github.io/cuda-to-tensor-core/lessons/27-methodology-wrap-up.html) | 12 | 实战方法论与收尾 | 少搬、多用、边搬边算 | [GIF](https://wikieden.github.io/cuda-to-tensor-core/animations/27-optimization-flow.gif) · [▶](https://wikieden.github.io/cuda-to-tensor-core/animations/27-optimization-flow.html) | C63–C64 | 正文完成，练习待做 |

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
