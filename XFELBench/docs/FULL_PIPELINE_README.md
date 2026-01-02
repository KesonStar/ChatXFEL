# XFELBench Full Evaluation Pipeline

完整的一键评估流程，自动生成多个配置、运行RAG评估并使用LLM作为评判器进行打分。

## 概览

本pipeline包含三个主要步骤：

1. **配置生成** (`generate_configs.py`) - 生成多个实验配置文件
2. **RAG评估** (`eval_generator.py`) - 使用不同配置生成答案
3. **LLM评判** (`llm_judge.py`) - 使用GPT-4o-mini对答案进行多维度评分

## 快速开始

### 最简单的方式：一键运行

```bash
# 设置OpenAI API密钥
export OPENAI_API_KEY="your-key-here"

# 运行完整pipeline（所有配置）
./run_all.sh
```

这将：
- 生成12个不同的配置文件
- 对每个配置运行RAG评估
- 使用LLM对所有结果进行评分
- 生成对比报告

### 使用Python脚本

```bash
# 运行所有配置
python run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json

# 只运行特定配置
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --configs baseline hybrid_search full_features

# 列出所有可用配置
python run_full_evaluation.py --list-configs
```

## 预定义配置

系统自动生成以下12个配置：

| 配置名称 | 描述 | 主要特性 |
|---------|------|---------|
| `baseline` | 基线配置 | Dense search + Reranking |
| `no_rerank` | 最简配置 | Dense search only |
| `hybrid_search` | 混合搜索 | Dense + Sparse (0.5/0.5) + Reranking |
| `hybrid_dense_heavy` | Dense偏重混合 | Dense + Sparse (0.7/0.3) |
| `hybrid_sparse_heavy` | Sparse偏重混合 | Dense + Sparse (0.3/0.7) |
| `query_rewrite` | 查询重写 | Baseline + Query rewriting |
| `routing` | 路由检索 | Baseline + Two-stage routing |
| `hybrid_rewrite` | 混合+重写 | Hybrid search + Query rewriting |
| `hybrid_routing` | 混合+路由 | Hybrid search + Routing |
| `full_features` | 全功能 | All features enabled |
| `rerank_top3` | Reranker精选 | Reranking with top_n=3 |
| `rerank_top10` | Reranker宽松 | Reranking with top_n=10 |

## 详细用法

### 1. 生成配置文件

```bash
# 生成所有配置
python generate_configs.py

# 生成特定配置
python generate_configs.py --configs baseline hybrid_search

# 列出所有可用配置模板
python generate_configs.py --list
```

配置文件将保存在 `configs/generated/` 目录。

### 2. 运行完整Pipeline

#### 基本用法

```bash
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --problem-set problem_sets/problem_set.md
```

#### 高级选项

```bash
# 跳过RAG生成（使用已有结果）
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --skip-generation

# 只运行RAG评估，跳过LLM打分
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --skip-llm-judge

# 指定OpenAI API密钥
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --api-key sk-xxx

# 运行特定配置
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --configs baseline hybrid_search full_features
```

### 3. 比较结果

```bash
# 显示所有配置的对比表格
python compare_results.py

# 生成CSV报告
python compare_results.py --csv comparison_report.csv

# 对比特定配置
python compare_results.py --compare baseline hybrid_search full_features

# 显示分数分布统计
python compare_results.py --stats
```

## 输出目录结构

```
XFELBench/
├── configs/
│   └── generated/              # 生成的配置文件
│       ├── baseline.yaml
│       ├── hybrid_search.yaml
│       └── ...
├── outputs/                    # RAG生成的答案
│   ├── 20250102_100000_baseline/
│   │   ├── results.jsonl      # 每个问题的答案和来源
│   │   └── summary.json       # 生成统计
│   └── ...
├── evaluations/                # LLM评判结果
│   ├── 20250102_100500_baseline/
│   │   ├── evaluation_results.jsonl  # 详细评分
│   │   └── evaluation_summary.json   # 汇总分数
│   └── summary_20250102_100500/      # 跨配置对比
│       ├── comparison.json
│       └── EVALUATION_REPORT.md      # 完整报告
```

## 评估维度

LLM评判器在三个维度上对每个答案进行1-5分评分：

### 1. Factual Accuracy（事实准确性）
- 评估答案中的事实和概念是否正确
- 与ground truth对比（如果有）
- 识别技术术语使用是否准确

### 2. Groundedness（证据支持度）
- 答案是否被检索到的文档支持
- 是否存在幻觉（未被文档支持的声明）
- 多个来源的整合质量

### 3. Coverage & Specificity（覆盖度和具体性）
- 是否全面回答问题
- 技术细节是否充分
- 是否过于宽泛或模糊

## 配置选项详解

每个配置文件包含以下可调参数：

### 模型设置
```yaml
model:
  llm_name: "Qwen3-30B-Instruct"  # 生成模型
  embedding_model: "BGE-M3"        # 嵌入模型
  temperature: 0.1                 # 温度参数
  num_predict: 2048                # 最大生成tokens
  num_ctx: 8192                    # 上下文窗口
```

### 检索特性
```yaml
features:
  query_rewrite:
    enabled: false                 # 查询重写
  hybrid_search:
    enabled: true                  # 混合搜索
    dense_weight: 0.5              # Dense权重
    sparse_weight: 0.5             # Sparse权重
  rerank:
    enabled: true                  # 重排序
    model: "BAAI/bge-reranker-v2-m3"
    top_n: 6                       # 重排后保留数量
  routing:
    enabled: false                 # 两阶段路由
    fulltext_top_k: 6              # 全文检索数量
```

### 检索参数
```yaml
retrieval:
  top_k: 10                        # 初始检索数量
  search_params:
    ef: 20                         # HNSW搜索参数
```

## 自定义配置

### 方法1：修改generate_configs.py

在 `EXPERIMENT_CONFIGS` 字典中添加新配置：

```python
"my_custom_config": {
    "name": "my_custom_config",
    "description": "My custom configuration",
    "features": {
        "query_rewrite": {"enabled": True},
        "hybrid_search": {
            "enabled": True,
            "dense_weight": 0.6,
            "sparse_weight": 0.4
        },
        "rerank": {
            "enabled": True,
            "model": "BAAI/bge-reranker-v2-m3",
            "top_n": 5
        },
        "routing": {"enabled": False},
        "chat_history": {"enabled": False}
    }
}
```

### 方法2：手动创建YAML文件

在 `configs/generated/` 目录创建新的YAML文件，参考现有配置格式。

## 性能优化建议

### 并行处理

目前pipeline是串行执行。如需并行处理多个配置：

```bash
# 手动并行运行（在不同终端）
python run_full_evaluation.py --configs baseline &
python run_full_evaluation.py --configs hybrid_search &
python run_full_evaluation.py --configs full_features &
wait
```

### 减少API成本

```bash
# 先运行所有RAG评估，不进行LLM打分
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --skip-llm-judge

# 稍后只对选定配置进行LLM打分
# （需要手动调用llm_judge.py）
```

### 快速测试

```bash
# 只测试少数配置
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --configs baseline no_rerank

# 使用小问题集进行测试
python run_full_evaluation.py \
    --questions problem_sets/test_small.json \
    --configs baseline
```

## 结果分析

### 查看排名

```bash
# 查看所有配置的排名
python compare_results.py
```

输出示例：
```
====================================================================================================
Configuration Ranking (by Overall Score)
====================================================================================================
Rank   Config                         Overall    Factual    Grounded   Coverage
----------------------------------------------------------------------------------------
1      full_features                  4.23       4.35       4.18       4.15
2      hybrid_rewrite                 4.18       4.30       4.12       4.12
3      hybrid_search                  4.05       4.15       3.98       4.02
...
```

### 生成报告

Pipeline自动生成markdown报告：

```bash
# 查看最新报告
cat evaluations/summary_*/EVALUATION_REPORT.md
```

报告包含：
- 配置排名表格
- 各维度最佳配置
- 详细配置说明
- 评估方法论

### 对比分析

```bash
# 详细对比两个配置
python compare_results.py --compare baseline full_features
```

输出差异：
```
Difference (Config 1 - Config 2):
  Overall:         -0.23
  Factual:         -0.20
  Groundedness:    -0.24
  Coverage:        -0.25
```

## 故障排除

### OpenAI API错误

```bash
# 检查API密钥
echo $OPENAI_API_KEY

# 测试API连接
python -c "from openai import OpenAI; client = OpenAI(); print('OK')"
```

### 配置生成失败

```bash
# 验证YAML语法
python -c "import yaml; yaml.safe_load(open('configs/generated/baseline.yaml'))"
```

### RAG评估错误

检查：
- Milvus连接是否正常
- Ollama服务是否运行
- 配置文件中的collection名称是否正确

```bash
# 测试Milvus连接
python -c "from utils import get_milvus_connection; print(get_milvus_connection())"
```

### 内存不足

如果处理大量问题时内存不足：

1. 减少batch_size
2. 分批运行不同配置
3. 使用 `--skip-generation` 重用已有结果

## 完整示例

### 场景1：评估所有配置

```bash
# 1. 设置API密钥
export OPENAI_API_KEY="sk-xxx"

# 2. 运行完整pipeline
./run_all.sh

# 3. 查看结果
python compare_results.py

# 4. 生成CSV报告
python compare_results.py --csv final_comparison.csv
```

### 场景2：快速测试新配置

```bash
# 1. 修改generate_configs.py添加新配置

# 2. 只生成该配置
python generate_configs.py --configs my_new_config

# 3. 运行评估
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --configs my_new_config

# 4. 与baseline对比
python compare_results.py --compare baseline my_new_config
```

### 场景3：重新评估已有结果

```bash
# 如果已经运行过RAG评估，只需重新打分

# 找到结果文件
ls outputs/

# 运行LLM评判
python llm_judge.py \
    --results outputs/20250102_100000_baseline/results.jsonl \
    --output evaluations/reeval_baseline \
    --problem-set problem_sets/problem_set.md
```

## 脚本参考

### run_all.sh

一键运行脚本，接受3个参数：

```bash
./run_all.sh [QUESTION_FILE] [PROBLEM_SET] [CONFIGS]

# 示例
./run_all.sh problem_sets/xfel_qa_basic.json problem_sets/problem_set.md all
./run_all.sh problem_sets/xfel_qa_basic.json problem_sets/problem_set.md "baseline hybrid_search"
```

### run_full_evaluation.py

主控脚本，支持完整选项：

```bash
python run_full_evaluation.py --help
```

### generate_configs.py

配置生成器：

```bash
python generate_configs.py --help
```

### compare_results.py

结果比较工具：

```bash
python compare_results.py --help
```

## 最佳实践

1. **先运行基线配置**：建立性能基准
2. **逐步添加特性**：了解每个特性的影响
3. **记录实验结果**：保存配置和对应分数
4. **使用版本控制**：跟踪配置变化
5. **定期备份结果**：评估结果很宝贵

## API成本估算

使用GPT-4o-mini进行评估的大致成本：

- 每个问题：3次API调用（三个维度）
- 每个配置50题：150次调用
- 12个配置：1800次调用
- 估计成本：$2-3（可能更低）

实际成本取决于：
- 答案长度
- 来源文档数量
- API定价（会变化）

## 贡献

欢迎贡献新的配置模板或改进！

## 支持

如有问题：
1. 查看本README
2. 检查 `LLM_JUDGE_README.md`
3. 查看 `QUICKSTART.md`
4. 联系开发团队

## 更新日志

- **2025-01-02**: 初始版本发布
  - 12个预定义配置
  - 完整pipeline自动化
  - 三维度LLM评判
  - 结果比较工具
