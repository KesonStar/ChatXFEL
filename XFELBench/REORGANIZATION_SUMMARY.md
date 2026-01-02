# XFELBench 文件重组总结

## 重组完成 ✅

XFELBench文件夹已经被重新组织为更清晰的结构。所有文件路径引用已更新。

## 新的目录结构

```
XFELBench/
├── README.md                    # 主文档（已更新）
├── REORGANIZATION_SUMMARY.md    # 本文件
│
├── bin/                         # 可执行脚本
│   ├── run_all.sh              # 一键运行完整pipeline ✅
│   ├── quick_test.sh           # 快速测试 ✅
│   └── example_evaluation.sh   # LLM评判示例 ✅
│
├── scripts/                     # Python脚本（新组织）
│   ├── evaluation/             # 评估相关
│   │   ├── eval_generator.py  # RAG答案生成器
│   │   ├── llm_judge.py       # LLM评判器
│   │   └── compare_results.py # 结果比较工具
│   ├── generation/             # 生成相关
│   │   ├── generate_configs.py # 配置生成器
│   │   └── test_generator.py   # 测试生成器
│   └── orchestration/          # 主控脚本
│       ├── run_full_evaluation.py  # 完整评估流程 ✅
│       └── analyze_results.py      # 结果分析
│
├── docs/                        # 文档
│   ├── FULL_PIPELINE_README.md # 完整pipeline文档
│   ├── LLM_JUDGE_README.md     # LLM评判器文档
│   └── FILES_CREATED.md        # 文件清单
│
├── configs/                     # 配置文件（保持不变）
│   ├── experiments/            # 手工创建的实验配置
│   └── generated/              # 自动生成的配置
│
├── problem_sets/                # 问题集（保持不变）
├── outputs/                     # RAG输出结果（保持不变）
├── evaluations/                 # LLM评判结果（保持不变）
└── prompts/                     # 提示词模板（保持不变）
```

## 主要变化

### 1. 脚本重新组织
- **之前**: 所有Python脚本散乱在根目录
- **之后**: 按功能分类到 `scripts/evaluation/`, `scripts/generation/`, `scripts/orchestration/`

### 2. Shell脚本集中
- **之前**: Shell脚本散乱在根目录
- **之后**: 统一放在 `bin/` 目录

### 3. 文档整理
- **之前**: 文档散乱在根目录
- **之后**: 集中在 `docs/` 目录

### 4. 路径引用更新
所有脚本中的路径引用已更新：

#### Shell脚本更新
- ✅ `bin/run_all.sh` - 更新为调用 `scripts/orchestration/run_full_evaluation.py`
- ✅ `bin/quick_test.sh` - 更新所有Python脚本路径
- ✅ `bin/example_evaluation.sh` - 更新为调用 `scripts/evaluation/llm_judge.py`

#### Python脚本更新
- ✅ `scripts/orchestration/run_full_evaluation.py` - 更新import和子脚本调用路径

## 如何使用

### 从XFELBench根目录运行

**重要**: 所有命令应从XFELBench根目录执行！

```bash
cd /path/to/ChatXFEL/XFELBench

# 一键运行
./bin/run_all.sh

# 快速测试
./bin/quick_test.sh

# 生成配置
python scripts/generation/generate_configs.py

# 运行评估
python scripts/orchestration/run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json

# 比较结果
python scripts/evaluation/compare_results.py
```

### 旧命令映射到新命令

| 旧命令 (之前) | 新命令 (现在) |
|-------------|--------------|
| `./run_all.sh` | `./bin/run_all.sh` |
| `./quick_test.sh` | `./bin/quick_test.sh` |
| `python generate_configs.py` | `python scripts/generation/generate_configs.py` |
| `python eval_generator.py` | `python scripts/evaluation/eval_generator.py` |
| `python llm_judge.py` | `python scripts/evaluation/llm_judge.py` |
| `python compare_results.py` | `python scripts/evaluation/compare_results.py` |
| `python run_full_evaluation.py` | `python scripts/orchestration/run_full_evaluation.py` |

## 验证重组 ✅

所有路径问题已修复！以下测试已通过：

```bash
cd /path/to/ChatXFEL/XFELBench

# 测试配置生成
python scripts/generation/generate_configs.py --list  ✅

# 测试完整pipeline脚本
python scripts/orchestration/run_full_evaluation.py --list-configs  ✅

# 生成测试配置
python scripts/generation/generate_configs.py --configs baseline  ✅
```

### 已修复的问题

1. ✅ 添加了 `__init__.py` 文件到所有 scripts 子目录
2. ✅ 修复了 `run_full_evaluation.py` 中的 import 路径
3. ✅ 修复了 `generate_configs.py` 中的输出目录路径
4. ✅ 修复了 `run_full_evaluation.py` 中的 base_dir 设置
5. ✅ 更新了所有 shell 脚本中的 Python 脚本路径
6. ✅ 修复了 `--list-configs` 参数解析

运行快速测试以验证所有功能正常：

```bash
cd /path/to/ChatXFEL/XFELBench
./bin/quick_test.sh
```

## 文档位置

- **主README**: `README.md` （在根目录，已更新）
- **完整pipeline文档**: `docs/FULL_PIPELINE_README.md`
- **LLM评判器文档**: `docs/LLM_JUDGE_README.md`
- **文件清单**: `docs/FILES_CREATED.md`
- **本文件**: `REORGANIZATION_SUMMARY.md` （重组说明）

## 好处

### 1. 更清晰的组织结构
- 按功能分类，易于查找
- 脚本、文档、配置分离

### 2. 更好的可维护性
- 相关文件集中管理
- 易于添加新功能

### 3. 更专业的项目结构
- 符合Python项目最佳实践
- 易于理解和协作

## 注意事项

1. **始终从XFELBench根目录运行命令**
2. 如果遇到路径错误，检查是否在正确的目录
3. 所有shell脚本应使用 `./bin/script_name.sh` 运行
4. 所有Python脚本应使用 `python scripts/category/script_name.py` 运行

## 下一步

可以继续使用新的组织结构：

```bash
# 设置API密钥
export OPENAI_API_KEY="your-key"

# 运行完整评估
./bin/run_all.sh

# 查看结果
python scripts/evaluation/compare_results.py
```

## 需要帮助？

查看文档：
- 快速开始: `README.md`
- 详细文档: `docs/FULL_PIPELINE_README.md`
- LLM评判: `docs/LLM_JUDGE_README.md`

---

**重组日期**: 2025-01-02
**状态**: ✅ 完成并验证
