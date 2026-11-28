# TruthfulRAG 模块：报告撰写与补充实验计划（2 天版）

> **分工**：**Sean** 负责方法分析，**Hakimi** 负责实验分析。
> 每人从分析 → 实验 → 作图 → 写作一条线做到底，只在 **D1 上午** 和 **D2 晚上** 两个时间点合并。
> 时间只有 2 天，所有任务分两档：**P0 必做**、**P1 有余力再做**。

本目录下的文件：

| 文件                    | 说明                                               |
| ----------------------- | -------------------------------------------------- |
| `plan.md`             | 本计划                                             |
| `trace_example.jsonl` | 追踪日志样例（三种方法各一行），即第四节的接口格式 |
| `cl_style.mplstyle`   | 共用的 matplotlib 作图样式                         |

---

## 一、已发现的问题（报告"自己见解"的主要来源）

| #  | 发现                                                                                                                                                                                                                                                                  | 位置                                                                                                          | 用途                                                            |
| -- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1  | **熵过滤规则与论文不一致**：论文写的是保留 ΔH > τ；代码实际是 ΔH ≥ 0 保留，ΔH < 0 时只要 `H_base / H_aug ≤ 10^τ` 也保留，再按 ΔH 取前 10 条；一条都不满足时保留熵最高的那条。规则很宽松，τ 也变成了对数比值阈值                                      | `TruthfulRAG/modules.py` 中的 `entropy_filter`                                                            | Sean 方法剖析的核心论点，E1/E2 用实验检验                       |
| 2  | **使用了标准答案信息**：计算熵时 `max_tokens = len(tokenize(sample["answer"]))`，即金标准答案长度                                                                                                                                                             | `modules.py` 中的 `entropy_filter`；`llm/backend.py` 中的 `cal_entroy`                                | 轻度标签泄漏，E2' 做去泄漏对照                                  |
| 3  | **路径打分与论文公式不一致**：论文是 α·实体覆盖率 + β·关系覆盖率；代码里没有 α/β，实际是实体名向量检索 + 答案类型关键词计数 + 关系向量检索的边投票                                                                                                        | `modules.py` 中的 `_build_query_context`；`utils.py` 中的 `cal_path_score_list` 和 `edge_vote_path` | 公式与实现的差距                                                |
| 4  | 熵按 top-10 logprob 归一化后用自然对数计算（论文写 log₂）；每条路径都要单独调用一次 LLM                                                                                                                                                                              | 同上                                                                                                          | 成本来源（附录中每题 36–62 秒）                                |
| 5  | 原文实验设计存疑：①显著性检验拿 FaithfulRAG 的单次结果和自己 10 次运行的均值比，不是配对检验，且温度为 0 却有方差；②CPR 称为"比例"，数值却大于 1，公式和文字的符号也不一致；③FaithfulRAG 在 Qwen-72B 上只有 5.31，很反常；④RealtimeQA 只有 113 条，1 题就是 0.88% | 论文表 3、表 5、附录                                                                                          | Hakimi 的实验评析                                               |
| 6  | ~~导入问题~~（**已更正**：包名本身没错，原判断有误。现已改成相对导入，从任何目录运行都能找到包，但这不算 bug 修复，报告里不要写）                                                                                                                              | —                                                                                                            | —                                                              |
| 7  | **最终生成同时包含原始全文**：`predict_answer` 的提示是 Facts（筛选后的路径）+ Context（完整原文），即 q⊕P⊕C，而非论文写的 q⊕P                                                                                                                             | `modules.py` 中的 `predict_answer`                                                                        | 方法实为"增强"而非"替换"；E3 用"保留路径不含答案时的准确率"检验 |
| 8  | **路径得分不参与排序**：计数只决定去留，路径按长度排序后由 `run()` 截取前 30 条，论文的 TopK(Ref) 不存在                                                                                                                                                      | `_build_query_context`、`utils.py` 中的 `remove_subpaths`                                               | 选择完全依赖熵过滤                                              |
| 9  | **评测指标偏宽松**：`acc_score` 判断的是"归一化后的预测是金标准的子串"，空预测也算对                                                                                                                                                                          | `evaluate.py`                                                                                               | 追踪日志同时记录 `correct`（官方）和 `em`（严格）           |
| 10 | **原代码的 asyncio 并发是假的**：OpenAI 客户端是同步调用，`gather` 实际串行。已改为 `asyncio.to_thread`，现在能真正并发                                                                                                                                     | `llm/openai.py`                                                                                             | 影响 Hakimi 的运行速度                                          |

数据（`datas/`）：FaithEval 1000 条、MuSiQue/SQuAD 的 golden 和 negative 各约 1770 条、RealtimeQA（timeqa_2022_nota）113 条。
golden 与 negative 文件按 `id` 一一对应：golden 的 `answer` 是真实答案，negative 的 `answer` 是被篡改后的反事实答案（例如 Tamaulipas → Sinaloa）。

---

## 二、报告框架

章节号以编译结果为准（本模块目前是第 2 章）。每一节的第一段概述本节内容；本组自主完成的分析与实验需在该段中写明。

```
2   TruthfulRAG：事实级知识冲突消解（章首段：本章内容与分工概述）
2.1 论文概述                                      —— Sean
    2.1.1 研究背景
    2.1.2 问题定义
    2.1.3 方法框架
    2.1.4 实验设置与主要结论
2.2 方法分析（不复述论文，只写实现对照与评述）    —— Sean
    2.2.1 实现与论文描述的对照
    2.2.2 方法的优势
    2.2.3 方法的局限
    2.2.4 待检验的研究假设（H1–H4，对应 2.4 各小节）
2.3 复现实验                                      —— Hakimi
    2.3.1 实验设置
    2.3.2 结果与对比
    2.3.3 原文实验设计评述
2.4 补充实验（节首段与共同设置已写好，在 main.tex 中）
    2.4.1 熵变信号的有效性（E1，H1）              —— Sean
    2.4.2 筛选策略比较（E2，H2）                  —— Sean
    2.4.3 流水线误差分析（E3，H3）                —— Sean
    2.4.4 外部知识错误下的鲁棒性（R3，H4）        —— Hakimi
    2.4.5 统计显著性与计算成本（R4）              —— Hakimi
2.5 局限与展望
    2.5.1 方法层面                                —— Sean
    2.5.2 实验层面                                —— Hakimi
2.6 小结                                          —— Hakimi
```

**写作规范**（两人统一）：
- 非必要不用双引号，少用加粗和括号
- 正文中不写代码出处（文件名、函数名），用"实现中"等表述
- 不用 → 等箭头符号，用文字表述流程
- 表格统一三线表（booktabs），`\small`，列用 `>{\raggedright\arraybackslash}p{宽度}`，`\arraystretch` 设为 1.35
- 引用节号时，2.x 称"节"，2.x.x 称"小节"

**LaTeX 组织**（已建好，`thesis.tex` 中 `\include{body/cl/main}`）：每个小节一个文件，文件头注明负责人，只改自己的文件。

- Sean：`body/cl/s1_overview.tex`、`s2_method.tex`、`s4a_probe_method.tex`、`s5a_discussion_method.tex`
- Hakimi：`body/cl/s3_reproduction.tex`、`s4b_probe_experiment.tex`、`s5b_discussion_experiment.tex`、`s6_summary.tex`（目前是占位文件，里面的 `\label` 已被其它小节引用，请保留）
- 待补内容用 `\cltodo{...}`（红字）；图片用 `\clfig{路径}{宽度}`，文件不存在时显示占位框，保证随时可编译
- 编译：在 `SEUCourseReportTexTemplate/` 下运行 `latexmk -xelatex thesis.tex`
- 先写中文，最后统一翻译成英文；图中的文字直接用英文，翻译时就不用重画
- 图片分别放 `figure/cl_method/`、`figure/cl_exp/`
- `bibfile.bib` 共用，只追加、不改别人的条目

---

## 三、实验设计（已按 2 天压缩）

### 统一设定

- 模型：**只用 qwen-flash**（DashScope，OpenAI 兼容接口）；这也是生成 trace 的模型。E2 的再生成同样使用 qwen-flash
- 样本：MuSiQue、SQuAD、FaithEval 各随机抽 **100 条**（固定 `seed=42`），RealtimeQA 用全部 113 条。MuSiQue/SQuAD 的 golden 和 negative 用**同一批 id**
- 并发：用 asyncio 并发（建议 8–16），否则 400 多题 × 约 36 秒/题，单线程跑不完
- 缓存：知识图谱、候选路径、每条路径的熵全部写进追踪日志（见第四节），后续换过滤策略时不再重算

### Sean：检验方法本身的假设

| 编号 | 优先级 | 内容                                                                                                                                                                 | 产出                           |
| ---- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| E1   | P0     | **熵升高是否代表"纠正性路径"**：读取每条候选路径的 ΔH，按路径是否包含金标准答案（字符串匹配）分成有用和无用两组，比较 ΔH 分布，并计算 ΔH 作为区分指标的 AUC | 小提琴图 + AUC 数值            |
| E2   | P0     | **过滤策略对比**（只重做最终生成）：论文规则 ΔH > τ、代码规则、全部保留、随机保留同样条数、按 ΔH 反向排序                                                   | 准确率对比表                   |
| E2'  | P1     | τ ∈ {0, 0.5, 1, 2, 3} 扫描；去泄漏对照（`max_tokens` 固定为 16）                                                                                                 | 准确率–τ 曲线                |
| E3   | P0     | **逐级漏斗**：答案在原始上下文中 → 在图里 → 在候选路径里 → 在过滤后的路径里 → 最终答对                                                                     | 漏斗图                         |
| E3'  | P1     | 从答错样本中人工抽 **30 条**，归到四类：三元组抽取错误、检索遗漏、过滤误删、生成错误                                                                           | 饼图或条形图 + 1–2 个典型案例 |

### Hakimi：检验结论能否成立、能否推广

| 编号 | 优先级 | 内容                                                                                                                                                                          | 产出                    |
| ---- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| R1   | P0     | **主结果复现**：w/o RAG、w/ RAG、TruthfulRAG 三种方法，四个数据集子集                                                                                                   | 与论文表 1 并排的对比表 |
| R2   | P0     | **非冲突场景复现**：MuSiQue-golden、SQuAD-golden 各 100 条，w/ RAG 和 TruthfulRAG 两种方法                                                                              | 与论文表 2 对比         |
| R3   | P0     | **外部知识有误**：用 R1 中 negative 数据的结果，改用真实答案（字段 `gold_true`）重新评估，看 TruthfulRAG 是否比 w/ RAG 更盲从错误的外部知识。**不需要额外运行** | 对比表                  |
| R3'  | P1     | **混合证据**：把 golden 和 negative 的上下文拼在一起，统计模型跟随哪一方的比例                                                                                          | 堆叠条形图              |
| R4   | P0     | **统计可靠性**：基于 R1 的逐样本结果做 McNemar 配对检验和 bootstrap 95% 置信区间                                                                                        | 显著性表                |
| R4'  | P1     | **成本收益**：每题的 LLM 调用次数、token 数、耗时，画"准确率 vs 成本"图；加跑 KRE 基线                                                                                  | 散点图                  |

---

## 四、接口约定：追踪日志（trace）

### 4.1 为什么需要这个接口

TruthfulRAG 每答一道题，要经过建图、检索、逐条路径算熵、最终生成，约 **10–15 次 LLM 调用、30–60 秒**。原始代码只输出最终答案，中间结果用完就丢了。

而两个人的实验需要的恰好是**同一次运行的不同部分**：

| 谁     | 实验               | 需要的数据                                                                                      |
| ------ | ------------------ | ----------------------------------------------------------------------------------------------- |
| Hakimi | R1/R2 准确率       | 每题的 `pred`、`correct`                                                                    |
| Hakimi | R3 外部知识有误    | 同一题的 `pred` 与 `gold_true`                                                              |
| Hakimi | R4 配对检验 / 成本 | 每题、每种方法的 `correct`，以及 `n_llm_calls`、`n_tokens`、`latency_s`                 |
| Sean   | E1 熵是否有效      | 每条候选路径的 `dH` 与 `has_answer`                                                         |
| Sean   | E2 过滤策略对比    | 每条候选路径的**完整文本**和 `dH`（用来换一种规则重新挑选）                             |
| Sean   | E3 漏斗            | `answer_in_context`、`graph.answer_in_graph`、候选路径与 `kept` 里是否有答案、`correct` |

如果各跑各的，TruthfulRAG 的在线运行要做两遍，2 天时间不够。所以约定：**TruthfulRAG 只跑一次，每道题把所有中间结果写成一行 JSON（JSONL 格式），两人都从这份文件里取数据。** 这份文件就是"接口"。字段一旦约定就不再改，两边就能各写各的分析脚本，互不等待。

### 4.2 谁做什么

```
 Sean（D1 上午）                  Hakimi（D1 中午起）                   两人（D1 晚起）
 ┌───────────────────┐          ┌──────────────────────────┐        ┌──────────────────────────┐
 │ 改代码：在流水线中 │  交付    │ 用批量脚本运行            │  产出  │ results/trace_*.jsonl     │
 │ 插入记录点，输出   │ ───────▶ │ 4 个数据集 × 3 种方法     │ ─────▶ │ Hakimi → R1~R4 统计       │
 │ trace JSONL        │          │ （+ golden 两个数据集）   │        │ Sean   → E1~E3 离线分析   │
 └───────────────────┘          └──────────────────────────┘        └──────────────────────────┘
```

1. **Sean 写（生产者代码）**：在 TruthfulRAG 里加记录点，D1 11:30 前交付，并用 3 条样本自测，确认字段齐全。
   - `make_knowledge_graph` 之后：记录 `graph.n_entities`、`graph.n_relations`、`graph.answer_in_graph`
   - `entropy_filter` 内部：记录 `H_base`，以及每条候选路径的 `text`、`H`、`dH`；再记录最终保留的 `kept`
   - `get_predictions` 之后：记录 `pred`、`correct`
   - 全程累计 `n_llm_calls`、`n_tokens`、`latency_s`
   - w/o RAG、w/ RAG 两个基线也按同样格式输出，只是图和路径相关字段为 `null`
2. **Hakimi 跑（唯一的在线运行方）**：用自己的批量脚本调用上述代码，负责并发、断点续跑、抽样；所有 trace 文件放到共享的 `results/` 目录。Hakimi 只调用、不改记录点；发现字段有问题就告诉 Sean 修改。
3. **两人复用（消费者）**：各自写分析脚本，只读 trace 文件，不再调用 API（E2 例外，见下）。

### 4.3 文件命名与字段

文件：`results/trace_{dataset}_{method}.jsonl`，每行一道题。

- `dataset` ∈ `faitheval` / `musique_neg` / `musique_gold` / `squad_neg` / `squad_gold` / `realtimeqa`
- `method` ∈ `wo_rag` / `w_rag` / `TruthfulRAG`

完整样例见 `trace_example.jsonl`（三种方法各一行）。字段说明：

| 字段                                         | 类型              | 含义                                                                           | 谁用               |
| -------------------------------------------- | ----------------- | ------------------------------------------------------------------------------ | ------------------ |
| `id`                                       | str               | 样本 id，与 `datas/` 一致，跨方法、跨 golden/negative **配对的主键**   | 两人               |
| `dataset`、`method`                      | str               | 同文件名                                                                       | 两人               |
| `question`、`choices`                    | str / list        | 原题；E2 重做生成时需要                                                        | Sean               |
| `gold`                                     | str               | 当前数据文件里的答案（negative 数据中是反事实答案）                            | 两人               |
| `gold_true`                                | str               | 真实答案；negative 数据从对应 golden 文件按 id 取，其余与 `gold` 相同        | Hakimi（R3）       |
| `pred`、`correct`                        | str / bool        | 模型答案（已从 JSON 中提取）、官方指标 `acc_score` 是否判对                  | 两人               |
| `pred_raw`、`em`                         | str / bool        | 模型原始输出、严格精确匹配（**新增字段**，见第一节第 9 条）              | 两人               |
| `answer_in_context`                        | bool              | 原始检索上下文里是否出现 `gold`                                              | Sean（E3）         |
| `graph`                                    | obj / null        | `n_entities`、`n_relations`、`answer_in_graph`                           | Sean（E3）         |
| `H_base`                                   | float / null      | 不给上下文时的答案熵                                                           | Sean（E1）         |
| `candidates`                               | list / null       | 每条候选路径：`idx`、`text`（完整路径文本）、`H`、`dH`、`has_answer` | Sean（E1、E2、E3） |
| `kept`                                     | list / null       | 原代码规则最终保留的候选路径 `idx`                                           | Sean（E2、E3）     |
| `n_llm_calls`、`n_tokens`、`latency_s` | int / int / float | 本题成本                                                                       | Hakimi（R4'）      |

### 4.4 怎么复用

**Hakimi：R1、R3、R4 都只读文件，不需要重跑。**

```python
import pandas as pd
df = pd.concat(pd.read_json(f, lines=True) for f in glob.glob("results/trace_*.jsonl"))

# R1：准确率表
df.groupby(["dataset", "method"])["correct"].mean()

# R3：negative 数据改用真实答案评估（不需要重跑）
neg = df[df.dataset.str.endswith("_neg")].copy()
neg["follow_true"] = neg.apply(lambda r: match(r.pred, r.gold_true), axis=1)

# R4：按 id 把两种方法对齐，做 McNemar
wide = df.pivot_table(index=["dataset", "id"], columns="method", values="correct")
```

**Sean：E1、E3 纯离线；E2 只重做最后一步生成。**

```python
cp = df[df.method == "TruthfulRAG"]

# E1：展开候选路径，比较 has_answer 两组的 dH 分布，算 AUC
cand = cp.explode("candidates").dropna(subset=["candidates"])
cand = pd.json_normalize(cand["candidates"])
roc_auc_score(cand["has_answer"], cand["dH"])

# E3：漏斗各级的比例
funnel = [cp.answer_in_context.mean(),
          cp.graph.map(lambda g: g["answer_in_graph"]).mean(),
          cp.candidates.map(lambda c: any(x["has_answer"] for x in c)).mean(),
          cp.apply(lambda r: any(r.candidates[i]["has_answer"] for i in r.kept), axis=1).mean(),
          cp.correct.mean()]

# E2：换一种规则重新挑路径，再用 question + 选出的 text 调一次 LLM 生成答案
#     （每题 1 次调用，比完整重跑便宜约 10 倍）
```

### 4.5 约定之外的规则

- D1 上午对齐会上确认字段后**不再改名、不再删字段**；确有需要只能**新增**字段
- `null` 表示该方法没有这一步，不要用空列表或 0 代替（避免把"没有"和"有但为空"混在一起）
- 作图统一使用 `cl_style.mplstyle`：脚本开头 `plt.style.use("SEUCourseReportTexTemplate/plan/cl_style.mplstyle")`；方法配色固定（见该文件注释），两人的图并排放时颜色含义一致

### 4.6 已交付的代码（Sean → Hakimi）

**单题接口**：`TruthfulRAG` 类新增方法 `trace_sample(sample, method, dataset_name, gold_true=None) -> dict`，返回值就是一行 trace。

```python
import os, json, asyncio
from TruthfulRAG import TruthfulRAG

def make_worker(i):   # 每个并发 worker 一个实例，各自独立的目录
    return TruthfulRAG(dataset=[], backend_type="openai", model_name="gpt-4o-mini",
                      similarity_model="all-MiniLM-L6-v2", threshold=1,
                      output_dir=f"./runs/out{i}", working_dir=f"./runs/kg{i}")

rec = await worker.trace_sample(sample, "TruthfulRAG", "musique_neg", gold_true=golden_by_id[sample["id"]]["answer"])
fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

**Hakimi 写批量脚本时必须注意**：

1. **一个实例同一时刻只能跑一道题**：实例内部只有一张图，每道题开始时都会清空重建。要并发就建多个实例（建议 4–8 个），每个实例的 `working_dir` 不同。
2. **`output_dir` 和 `working_dir` 必须显式传入**：默认值里带冒号（时间戳），在 Windows 上建目录会失败。
3. 基线方法 `wo_rag` / `w_rag` 的调用开销很小，可以开更高的并发。
4. `trace_sample` 不捕获异常，由批量脚本决定重试和记录；建议按 `(dataset, method, id)` 断点续跑。
5. 写完后运行 `python experiments/trace_schema.py "results/trace_*.jsonl"` 校验格式。

**改动清单**（全部是附加改动，原有的 `run()` / `run_baseline()` 行为不变）：

- `TruthfulRAG/trace.py`（新增）：每题的调用次数和 token 计数（基于 ContextVar，并发时互不干扰），以及"文本是否含答案"的判定
- `TruthfulRAG/modules.py`：`entropy_filter` 新增可选参数 `return_details`
- `TruthfulRAG/pipeline.py`：新增 `trace_sample`；导入改为相对导入
- `TruthfulRAG/llm/openai.py`：改用 `asyncio.to_thread` 实现真正的并发，并记录 token 用量；`llm/hf.py` 记录调用次数
- `TruthfulRAG/kg/*.py`：导入改为相对导入

**测试与分析脚本**：

- `experiments/tests/offline_trace_smoke.py`：离线冒烟测试
- `experiments/tests/make_mock_traces.py`：trace（写到 `experiments/_mock/`，已加入 gitignore）
- `experiments/analysis/e1_entropy.py`、`e2_filters.py`、`e3_funnel.py`：E1–E3，默认读取 `results/trace_*.jsonl`，图输出到 `figure/cl_method/`，表输出到 `experiments/out/cl_method/`

---

## 五、分工与两天节奏

### 分工总览

|      | Sean                                                | Hakimi                                               |
| ---- | --------------------------------------------------- | ---------------------------------------------------- |
| 工程 | 修导入问题，在流水线中加入 trace 记录点（D1 上午）  | 写批量运行脚本、管理并发和断点续跑，负责所有在线运行 |
| 实验 | E1、E2、E3（以离线分析为主，E2 需重做一次最终生成） | R1、R2、R3、R4（以运行和统计检验为主）               |
| 写作 | X.1、X.2、X.4.1–4.2、X.5 前半                      | X.3、X.4.3–4.4、X.5 后半、X.6                       |
| 整合 | 统一术语和符号表                                    | 维护作图样式，负责最终合并和编译                     |

工作量：Sean 的写作量更大（方法剖析加论文/代码对照），在线运行少；Hakimi 的运行和调试多，写作偏数据描述。两边大致持平。

### D1

| 时段                  | Sean                                                                                                                                          | Hakimi                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 上午（约 3h）         | **09:00–09:30 两人对齐**：确认框架、trace 字段、样本 id 列表。之后修导入问题，加 trace 记录点，用 3 条样本自测，**11:30 前交付** | 写批量运行脚本（并发、断点续跑、抽样并导出 id 列表）；起草 X.3.1 复现设置                |
| 下午（约 4h）         | 写 X.1、X.2.1–X.2.3（不依赖实验结果）；用 `trace_example.jsonl` 和自测数据开发 E1/E3 的分析和作图脚本                                      | **12:00 启动 R1**（四个数据集 × 三种方法，后台运行），再启动 R2；等待期间写 X.3.3 |
| 晚上（约 3h）         | TruthfulRAG 的 trace 文件一出来就跑 E1、E3；启动 E2 的重新生成                                                                                | 检查 R1/R2 结果，整理与论文的对比表；完成 R3（只读文件）                                 |
| **D1 晚检查点** | 互发一次进度：R1 是否跑完、E1 是否有初步结论。若进度落后，P1 全部取消                                                                         |                                                                                          |

### D2

| 时段                        | Sean                                              | Hakimi                                                    |
| --------------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| 上午                        | 汇总 E2 结果；有余力做 E2'、E3'（人工归因 30 条） | 完成 R4（McNemar + bootstrap）；有余力做 R3'、R4'         |
| 下午                        | 写 X.2.4、X.4.1、X.4.2，定稿图表                  | 写 X.3.2、X.4.3、X.4.4，定稿图表                          |
| 傍晚                        | 写 X.5 前半（方法层面的局限与改进）               | 写 X.5 后半（实验层面的局限）、X.6                        |
| **晚上合并（约 2h）** | 审读 Hakimi 的部分；统一术语和符号                | 审读 Sean 的部分；合并 `.tex`、编译、检查图表编号和引用 |

### 风险与预案

- **API 跑得慢或限流**：先保 RealtimeQA（113 条）和 MuSiQue（100 条）跑完，再跑另外两个；必要时把子集降到 50 条，并在报告中说明
- **trace 记录点延误**：Hakimi 先跑 w/o RAG 和 w/ RAG（这两种方法的 trace 只需在脚本里直接写出，不依赖 Sean 的改动），TruthfulRAG 等 Sean 交付后再跑
- **结果与论文差距大**：不追求对齐，如实报告，结合第一节第 1–4 条的实现差异分析原因，这本身就是有价值的内容
