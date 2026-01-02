# XFELBench - XFEL RAG Evaluation Benchmark

完整的RAG系统评估框架，支持自动配置生成、批量评估和LLM评判打分。

## 快速开始

```bash
# 1. 设置OpenAI API密钥
export OPENAI_API_KEY="your-key-here"

# 2. 运行完整评估pipeline（推荐）
./bin/run_all.sh

# 3. 或者快速测试
./bin/quick_test.sh
```

## 目录结构

```
XFELBench/
├── README.md                    # 本文件
├── bin/                         # 可执行脚本
│   ├── run_all.sh              # 一键运行完整pipeline
│   ├── quick_test.sh           # 快速测试
│   └── example_evaluation.sh   # LLM评判示例
├── scripts/                     # Python脚本
│   ├── evaluation/             # 评估相关
│   │   ├── eval_generator.py  # RAG答案生成器
│   │   ├── llm_judge.py       # LLM评判器
│   │   └── compare_results.py # 结果比较工具
│   ├── generation/             # 生成相关
│   │   ├── generate_configs.py # 配置生成器
│   │   └── test_generator.py   # 测试生成器
│   └── orchestration/          # 主控脚本
│       ├── run_full_evaluation.py  # 完整评估流程
│       └── analyze_results.py      # 结果分析
├── docs/                        # 文档
│   ├── FULL_PIPELINE_README.md # 完整pipeline文档
│   ├── LLM_JUDGE_README.md     # LLM评判器文档
│   └── FILES_CREATED.md        # 文件清单
├── configs/                     # 配置文件
│   ├── experiments/            # 实验配置
│   └── generated/              # 自动生成的配置
├── problem_sets/                # 问题集
├── outputs/                     # RAG输出结果
├── evaluations/                 # LLM评判结果
└── prompts/                     # 提示词模板
```

## 主要功能

### 1. 自动配置生成

生成12个预定义的RAG配置，涵盖不同的检索策略：

```bash
python scripts/generation/generate_configs.py

# 列出所有可用配置
python scripts/generation/generate_configs.py --list
```

**可用配置**:
- `baseline` - 基线（Dense + Reranking）
- `hybrid_search` - 混合搜索（Dense + Sparse）
- `query_rewrite` - 查询重写
- `routing` - 两阶段路由
- `full_features` - 全功能
- 等等...共12个配置

### 2. RAG评估

对每个配置生成答案：

```bash
python scripts/evaluation/eval_generator.py \
    --config configs/generated/baseline.yaml \
    --questions problem_sets/xfel_qa_basic.json
```

### 3. LLM评判

使用GPT-4o-mini进行三维度评分：

```bash
python scripts/evaluation/llm_judge.py \
    --results outputs/20251230_230056_baseline/results.jsonl \
    --output evaluations/baseline_eval \
    --problem-set problem_sets/problem_set.md
```

**评估维度**:
- Factual Accuracy (1-5)
- Groundedness / Evidence Use (1-5)
- Coverage & Specificity (1-5)

### 4. 完整Pipeline

一键运行所有配置的评估和打分：

```bash
python scripts/orchestration/run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json

# 或使用shell脚本
./bin/run_all.sh
```

### 5. 结果比较

对比不同配置的表现：

```bash
python scripts/evaluation/compare_results.py

# 生成CSV报告
python scripts/evaluation/compare_results.py --csv results.csv

# 对比特定配置
python scripts/evaluation/compare_results.py --compare baseline hybrid_search
```

## 使用示例

### 示例1：完整评估所有配置

```bash
export OPENAI_API_KEY="sk-xxx"
./bin/run_all.sh
```

### 示例2：评估特定配置

```bash
python scripts/orchestration/run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --configs baseline hybrid_search full_features
```

### 示例3：自定义配置

1. 编辑 `scripts/generation/generate_configs.py`
2. 添加你的配置到 `EXPERIMENT_CONFIGS`
3. 生成并运行：

```bash
python scripts/generation/generate_configs.py --configs my_config
python scripts/orchestration/run_full_evaluation.py --configs my_config
```

## 输出说明

### RAG输出 (`outputs/`)

```json
{
  "question_id": "basic_001",
  "question": "What is SFX?",
  "answer": "Serial Femtosecond Crystallography...",
  "sources": [...],
  "generation_time": 15.2
}
```

### 评判输出 (`evaluations/`)

```json
{
  "evaluation": {
    "factual_accuracy": {"score": 5, "reasoning": "..."},
    "groundedness": {"score": 4, "reasoning": "..."},
    "coverage_specificity": {"score": 5, "reasoning": "..."},
    "average_score": 4.67
  }
}
```

### 汇总报告 (`evaluations/summary_*/`)

- `comparison.json` - 所有配置的分数对比
- `EVALUATION_REPORT.md` - 完整markdown报告

## 命令速查

```bash
# 列出可用配置
python scripts/generation/generate_configs.py --list

# 生成配置
python scripts/generation/generate_configs.py

# 运行评估（所有配置）
./bin/run_all.sh

# 快速测试（2个配置）
./bin/quick_test.sh

# 比较结果
python scripts/evaluation/compare_results.py

# 查看最新报告
cat $(ls -t evaluations/summary_*/EVALUATION_REPORT.md | head -1)
```

## 详细文档

- **完整Pipeline**: `docs/FULL_PIPELINE_README.md`
- **LLM评判器**: `docs/LLM_JUDGE_README.md`
- **配置路径指南**: `CONFIG_PATH_GUIDE.md` ⭐ 重要
- **导入路径修复**: `IMPORT_FIX_SUMMARY.md`
- **文件清单**: `docs/FILES_CREATED.md`

## 依赖

- Python 3.10+
- OpenAI API (GPT-4o-mini)
- PyYAML
- 其他依赖见主项目 `requirements.txt`

## API成本

使用GPT-4o-mini评估：
- 每个问题约 $0.01-0.02
- 50题×12配置 ≈ $6-12

## 故障排除

### API密钥错误

```bash
export OPENAI_API_KEY="your-key"
```

### 路径问题

所有脚本应从XFELBench根目录运行：

```bash
cd /path/to/ChatXFEL/XFELBench
./bin/run_all.sh
```

### 导入错误

确保Python能找到ChatXFEL模块：

```bash
# 从XFELBench根目录运行所有脚本
cd /path/to/ChatXFEL/XFELBench
python scripts/orchestration/run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json
```

## 贡献

欢迎贡献新的配置模板、评估维度或改进！

## 许可

与ChatXFEL主项目相同

## 联系

如有问题，请查看文档或联系开发团队。

---

**提示**: 首次使用建议先运行 `./bin/quick_test.sh` 验证设置！
