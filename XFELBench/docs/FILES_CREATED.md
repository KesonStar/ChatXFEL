# 创建的文件清单

本次为XFELBench创建的完整评估pipeline包含以下文件：

## 核心脚本

### 1. `llm_judge.py` ⭐
**功能**: LLM评判器，使用GPT-4o-mini对RAG答案进行三维度评分

**用法**:
```bash
python llm_judge.py \
    --results outputs/20251230_230056_baseline/results.jsonl \
    --output evaluations/baseline_eval \
    --problem-set problem_sets/problem_set.md
```

**评估维度**:
- Factual Accuracy (1-5分)
- Groundedness / Evidence Use (1-5分)
- Coverage & Specificity (1-5分)

**输出**:
- `evaluation_results.jsonl` - 每个问题的详细评分
- `evaluation_summary.json` - 汇总统计

---

### 2. `generate_configs.py` ⭐
**功能**: 自动生成多个实验配置文件

**用法**:
```bash
# 生成所有12个配置
python generate_configs.py

# 生成特定配置
python generate_configs.py --configs baseline hybrid_search

# 列出可用配置
python generate_configs.py --list
```

**生成的配置**:
- `baseline` - 基线配置
- `no_rerank` - 无重排序
- `hybrid_search` - 混合搜索
- `hybrid_dense_heavy` - Dense偏重
- `hybrid_sparse_heavy` - Sparse偏重
- `query_rewrite` - 查询重写
- `routing` - 路由检索
- `hybrid_rewrite` - 混合+重写
- `hybrid_routing` - 混合+路由
- `full_features` - 全功能
- `rerank_top3` - 重排Top3
- `rerank_top10` - 重排Top10

**输出位置**: `configs/generated/`

---

### 3. `run_full_evaluation.py` ⭐⭐⭐
**功能**: 主控脚本，orchestrate完整的评估pipeline

**流程**:
1. 生成配置文件
2. 对每个配置运行RAG评估
3. 对每个结果运行LLM评判
4. 生成对比报告

**用法**:
```bash
# 运行所有配置
python run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json

# 运行特定配置
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --configs baseline hybrid_search full_features

# 跳过LLM评判（仅生成答案）
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --skip-llm-judge
```

**输出**:
- RAG结果: `outputs/TIMESTAMP_CONFIG/`
- 评估结果: `evaluations/TIMESTAMP_CONFIG/`
- 总结报告: `evaluations/summary_TIMESTAMP/`

---

### 4. `compare_results.py`
**功能**: 比较和可视化不同配置的评估结果

**用法**:
```bash
# 显示排名表格
python compare_results.py

# 生成CSV报告
python compare_results.py --csv comparison.csv

# 对比特定配置
python compare_results.py --compare baseline hybrid_search full_features

# 显示统计信息
python compare_results.py --stats
```

**功能特性**:
- 排名表格
- 各维度最佳配置
- 分数分布统计
- 详细对比分析
- CSV导出

---

## Shell脚本

### 5. `run_all.sh` ⭐⭐⭐
**功能**: 一键运行脚本，最简单的使用方式

**用法**:
```bash
# 设置API密钥
export OPENAI_API_KEY="your-key"

# 运行
./run_all.sh
```

**参数** (可选):
```bash
./run_all.sh [QUESTION_FILE] [PROBLEM_SET] [CONFIGS]
```

---

### 6. `quick_test.sh`
**功能**: 快速测试脚本，验证pipeline设置

**用法**:
```bash
./quick_test.sh
```

**测试内容**:
- 生成2个配置（baseline, hybrid_search）
- 运行RAG评估
- 运行LLM评判（如果API key可用）
- 显示对比结果

---

### 7. `example_evaluation.sh`
**功能**: LLM judge的示例脚本

**用法**:
```bash
# 修改脚本中的OPENAI_API_KEY
./example_evaluation.sh
```

---

## 文档

### 8. `FULL_PIPELINE_README.md` ⭐⭐
**内容**: 完整pipeline的详细使用文档

**包含**:
- 快速开始指南
- 所有配置说明
- 详细用法示例
- 故障排除
- 最佳实践
- API成本估算

---

### 9. `LLM_JUDGE_README.md`
**内容**: LLM评判器的详细文档

**包含**:
- 评估维度说明
- 评分标准
- 使用示例
- Ground truth处理
- API配置
- 编程接口

---

### 10. `FILES_CREATED.md` (本文件)
**内容**: 所有创建文件的清单和说明

---

## 使用流程

### 方案A: 一键运行（推荐初次使用）

```bash
# 1. 设置API密钥
export OPENAI_API_KEY="sk-xxx"

# 2. 运行
./run_all.sh

# 3. 查看结果
python compare_results.py
```

### 方案B: 分步运行（推荐调试和自定义）

```bash
# 1. 生成配置
python generate_configs.py

# 2. 运行完整评估
python run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json

# 3. 比较结果
python compare_results.py --csv results.csv
```

### 方案C: 自定义配置

```bash
# 1. 修改generate_configs.py，添加自定义配置

# 2. 生成该配置
python generate_configs.py --configs my_custom_config

# 3. 运行评估
python run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json \
    --configs my_custom_config

# 4. 与baseline对比
python compare_results.py --compare baseline my_custom_config
```

### 方案D: 仅运行LLM评判

如果已经有RAG结果：

```bash
python llm_judge.py \
    --results outputs/20251230_230056_baseline/results.jsonl \
    --output evaluations/my_eval \
    --problem-set problem_sets/problem_set.md
```

---

## 快速参考

### 查看帮助

```bash
python run_full_evaluation.py --help
python generate_configs.py --help
python llm_judge.py --help
python compare_results.py --help
```

### 列出配置

```bash
python run_full_evaluation.py --list-configs
python generate_configs.py --list
```

### 查看结果

```bash
# 最新的评估报告
ls -t evaluations/summary_*/EVALUATION_REPORT.md | head -1

# 查看报告
cat $(ls -t evaluations/summary_*/EVALUATION_REPORT.md | head -1)
```

### 清理

```bash
# 清理生成的配置
rm -rf configs/generated/

# 清理输出（谨慎！）
rm -rf outputs/
rm -rf evaluations/
```

---

## 文件依赖关系

```
run_all.sh
    ↓
run_full_evaluation.py
    ↓
    ├─→ generate_configs.py → configs/generated/*.yaml
    │
    ├─→ eval_generator.py → outputs/*/results.jsonl
    │
    └─→ llm_judge.py → evaluations/*/evaluation_results.jsonl
            ↓
        compare_results.py
```

---

## 关键特性

### ✅ 已实现

- [x] 自动生成12个预定义配置
- [x] 批量RAG评估
- [x] 三维度LLM评判
- [x] 条件性ground truth处理
- [x] 结果对比和排名
- [x] Markdown报告生成
- [x] CSV导出
- [x] 错误处理和重试
- [x] 进度跟踪
- [x] 完整文档

### 🔄 可扩展

- [ ] 并行处理多个配置
- [ ] 更多评估维度
- [ ] 可视化图表
- [ ] 统计显著性检验
- [ ] 实时进度监控
- [ ] 结果缓存

---

## 预期输出示例

### 1. 配置排名表格

```
====================================================================================================
Configuration Ranking (by Overall Score)
====================================================================================================
Rank   Config                         Overall    Factual    Grounded   Coverage   Total Q
----------------------------------------------------------------------------------------
1      full_features                  4.23       4.35       4.18       4.15       50
2      hybrid_rewrite                 4.18       4.30       4.12       4.12       50
3      hybrid_search                  4.05       4.15       3.98       4.02       50
4      baseline                       3.95       4.05       3.88       3.92       50
...
```

### 2. 最佳配置

```
Best Configurations by Dimension
====================================================================================================
Dimension                 Configuration                  Score
-----------------------------------------------------------------
Overall                   full_features                  4.23
Factual Accuracy          full_features                  4.35
Groundedness              hybrid_routing                 4.20
Coverage & Specificity    query_rewrite                  4.18
```

### 3. 评估报告

自动生成的markdown报告包含：
- 完整排名表格
- 各维度分析
- 配置详情
- 方法论说明

---

## 技术栈

- **Python 3.10+**
- **OpenAI API** (GPT-4o-mini)
- **YAML** 配置文件
- **JSONL** 结果存储
- **Bash** 脚本
- **Markdown** 报告

---

## 成本估算

### API调用

- 每个问题：3次调用（三个维度）
- 50个问题：150次调用
- 12个配置：1800次调用

### OpenAI成本

使用GPT-4o-mini:
- 输入: ~$0.15/1M tokens
- 输出: ~$0.60/1M tokens
- **估计总成本**: $2-5（取决于答案长度）

---

## 支持与反馈

如有问题或建议：

1. 查看 `FULL_PIPELINE_README.md`
2. 查看 `LLM_JUDGE_README.md`
3. 运行 `quick_test.sh` 诊断问题
4. 联系开发团队

---

## 版本信息

- **创建日期**: 2025-01-02
- **版本**: 1.0
- **兼容性**: XFELBench 1.0+

---

## 总结

本评估pipeline提供了：

1. **自动化**: 一键运行完整评估
2. **灵活性**: 12个预定义配置 + 自定义选项
3. **全面性**: 三维度评分 + 详细分析
4. **可用性**: 清晰的文档和示例
5. **可扩展性**: 易于添加新配置和评估维度

**推荐入门方式**: 先运行 `./quick_test.sh` 测试设置，然后运行 `./run_all.sh` 完整评估。
