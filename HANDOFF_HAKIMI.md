# Hakimi 交接说明（实验线）

你负责：**所有在线运行** + R1–R4 + 报告 X.3、X.4.3、X.4.4、X.5 后半、X.6。
Sean 已经把"带记录的流水线"做好了，你只需要写一个批量脚本调用它。完整计划见
[`SEUCourseReportTexTemplate/plan/plan.md`](SEUCourseReportTexTemplate/plan/plan.md)，
本文只列你**现在**要做的事。

## 1. 先看这三样

| 文件 | 看什么 |
|---|---|
| `SEUCourseReportTexTemplate/plan/plan.md` 第四节 | trace 接口：为什么要有、字段含义、**4.6 节是调用方法和注意事项** |
| `SEUCourseReportTexTemplate/plan/trace_example.jsonl` | 三种方法各一行的样例 |
| `experiments/trace_schema.py` | 格式校验，写完后用它检查 |

## 2. 环境

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # PowerShell: $env:OPENAI_API_KEY="..."
export OPENAI_BASE_URL=...       # 用官方接口可不设
```
首次运行会下载 `all-MiniLM-L6-v2`。Sean 没有在真实 API 下跑过，**你的第一次试跑就是真实环境冒烟测试**。

## 3. 样本清单（已固定，别重新抽样）

`experiments/samples/ids_{dataset}.txt`，每行一个 id，seed=42。
MuSiQue / SQuAD 的 neg 和 gold 用同一批 id。重新生成：`python experiments/make_samples.py`。
数据在 `datas/`：用 id 取样本。`gold_true` 取法：negative 数据的真实答案 = 对应 golden 文件中同 id 的 `answer`；其它数据集等于 `answer`。

## 4. 你要写的批量脚本

对每个 `(dataset, method)` 的每个 id 调用 `trace_sample`，一行写入 `results/trace_{dataset}_{method}.jsonl`。
- 方法：`wo_rag`、`w_rag`、`TruthfulRAG`；数据集：`faitheval`、`musique_neg`、`musique_gold`、`squad_neg`、`squad_gold`、`realtimeqa`。
- **一个实例同一时刻只能跑一道题**，要并发就开 4–8 个实例，各自独立的 `working_dir`。
- `output_dir`、`working_dir` 必须显式传（默认值带冒号，Windows 下建目录会失败）。
- 按 `(dataset, method, id)` 断点续跑；异常自己捕获并重试，`trace_sample` 不吞异常。
- 基线方法开销小，并发可以更高。

## 5. 顺序建议

1. **先试跑 3 道题 × 3 种方法**，用 `python experiments/trace_schema.py "results/trace_*.jsonl"` 校验。
   重点看：候选路径是否非空；`n_llm_calls` 是否约为 4 + 路径数（每个文本块 2 次建图调用）；单题耗时是否在几十秒量级。**出问题立刻告诉 Sean。**
2. 先跑 `wo_rag`、`w_rag`（快、便宜，不依赖复杂逻辑）。
3. 再跑 `TruthfulRAG`：先 `realtimeqa` 和 `musique_neg`，再其它。
4. 跑完后告诉 Sean，他开始 E1/E3；E2 需要他自己的 API 环境，和你互不影响。

## 6. 报告怎么写

- 你的文件在 `SEUCourseReportTexTemplate/body/cl/`：`s3_reproduction.tex`、`s4b_probe_experiment.tex`、`s5b_discussion_experiment.tex`、`s6_summary.tex`。**只改这几个**，里面的 `\label` 已被其它小节引用，请保留。
- 待补用 `\cltodo{...}`，图用 `\clfig{路径}{宽度}`（文件不存在时显示占位框）。
- 编译：在 `SEUCourseReportTexTemplate/` 下 `latexmk -xelatex thesis.tex`。
- 图用 `SEUCourseReportTexTemplate/plan/cl_style.mplstyle`，放到 `figure/cl_exp/`；图中文字用英文。
- 先写中文，最后统一翻译。

## 7. 分析时可直接参考

`experiments/trace_schema.py` 里的 `load_traces("results/trace_*.jsonl")` 一次读入所有文件，返回 DataFrame，并检查 `(dataset, method, id)` 是否重复。
R3 只读文件即可：用 `gold_true` 重新评估 negative 数据的预测，不用额外运行。
