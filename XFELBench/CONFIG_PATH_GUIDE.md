# 配置文件路径指南

## 概述

XFELBench配置文件中的路径设置需要特别注意，因为不同的配置项有不同的基础目录。

## 配置文件结构

```yaml
prompt:
  template_file: "prompts/naive.pt"  # ← 相对于 ChatXFEL 根目录

collection:
  name: "xfel_bibs_collection_with_abstract"  # ← Milvus集合名，不是路径

database:
  milvus:
    host: "10.19.48.181"  # ← 网络地址，不是路径
```

## 路径基础目录规则

### 1. Prompt模板路径 ⭐ 重要

**配置项**: `prompt.template_file`

**基础目录**: ChatXFEL根目录 (`/path/to/ChatXFEL/`)

**示例**:
```yaml
prompt:
  template_file: "prompts/naive.pt"
```

**实际路径**: `/path/to/ChatXFEL/prompts/naive.pt`

**可用的prompt模板**:
- `prompts/naive.pt` - 简单的问答prompt
- `prompts/chat_with_history.pt` - 带历史记录的对话prompt

**为什么这样设计**:
- Prompt模板是ChatXFEL主项目的一部分
- 所有RAG功能都依赖于主项目的`rag.py`
- Prompt模板存储在主项目的`prompts/`目录

### 2. 问题集路径

**在命令行指定**: `--questions problem_sets/xfel_qa_basic.json`

**基础目录**: XFELBench根目录 (`/path/to/ChatXFEL/XFELBench/`)

**示例**:
```bash
python scripts/orchestration/run_full_evaluation.py \
    --questions problem_sets/xfel_qa_basic.json
```

**实际路径**: `/path/to/ChatXFEL/XFELBench/problem_sets/xfel_qa_basic.json`

### 3. 输出目录

**自动生成**: `outputs/TIMESTAMP_CONFIG_NAME/`

**基础目录**: XFELBench根目录 (`/path/to/ChatXFEL/XFELBench/`)

**示例**:
```
outputs/20250102_153000_baseline/
├── config.yaml
├── results.jsonl
└── summary.json
```

### 4. 评估结果目录

**自动生成**: `evaluations/TIMESTAMP_CONFIG_NAME/`

**基础目录**: XFELBench根目录 (`/path/to/ChatXFEL/XFELBench/`)

**示例**:
```
evaluations/20250102_153500_baseline/
├── evaluation_results.jsonl
└── evaluation_summary.json
```

### 5. 配置文件目录

**自动生成**: `configs/generated/`

**基础目录**: XFELBench根目录 (`/path/to/ChatXFEL/XFELBench/`)

**示例**:
```
configs/generated/
├── baseline.yaml
├── hybrid_search.yaml
└── full_features.yaml
```

## 目录结构总览

```
ChatXFEL/                           # ChatXFEL根目录
├── prompts/                        # ← Prompt模板在这里 ⭐
│   ├── naive.pt
│   └── chat_with_history.pt
├── rag.py                          # RAG核心模块
├── utils.py
├── query_rewriter.py
└── XFELBench/                      # XFELBench根目录
    ├── configs/                    # ← 配置文件
    │   ├── experiments/
    │   └── generated/
    ├── problem_sets/               # ← 问题集
    ├── outputs/                    # ← RAG输出
    ├── evaluations/                # ← 评估结果
    └── scripts/
        └── evaluation/
            └── eval_generator.py   # 在这里解析prompt路径
```

## 常见错误和解决方案

### 错误1: Prompt文件未找到

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory:
'/Users/.../XFELBench/prompts/naive.pt'
```

**原因**: 试图在XFELBench目录下找prompt文件

**解决方案**:
- ✅ 确认配置文件中 `template_file: "prompts/naive.pt"` (相对路径)
- ✅ 确认 `eval_generator.py` 中使用 `CHATXFEL_ROOT` 来解析路径
- ✅ 确认 `/path/to/ChatXFEL/prompts/naive.pt` 文件存在

### 错误2: 问题集文件未找到

**错误信息**:
```
FileNotFoundError: problem_sets/xfel_qa_basic.json not found
```

**原因**: 从错误的目录运行命令

**解决方案**:
```bash
# 必须从XFELBench目录运行
cd /path/to/ChatXFEL/XFELBench
python scripts/orchestration/run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json
```

### 错误3: 配置文件路径错误

**错误信息**:
```
FileNotFoundError: configs/generated/baseline.yaml not found
```

**原因**: 配置文件还未生成

**解决方案**:
```bash
# 先生成配置
python scripts/generation/generate_configs.py

# 然后运行评估
./bin/run_all.sh
```

## 绝对路径 vs 相对路径

### 使用相对路径（推荐）

✅ **推荐配置**:
```yaml
prompt:
  template_file: "prompts/naive.pt"  # 相对于ChatXFEL根目录
```

**优点**:
- 项目可移植
- 在不同机器上都能工作
- 配置文件可以共享

### 使用绝对路径（不推荐）

❌ **不推荐**:
```yaml
prompt:
  template_file: "/Users/kesonstar/Desktop/SHTECH/G4S1/CS286/ChatXFEL/prompts/naive.pt"
```

**缺点**:
- 不可移植
- 每个用户都需要修改
- 配置文件无法共享

**何时使用**: 仅当prompt模板在非标准位置时

## 自定义Prompt模板

如果你想使用自定义的prompt模板：

### 方法1: 放在ChatXFEL/prompts/目录（推荐）

```bash
# 1. 创建自定义prompt
vim /path/to/ChatXFEL/prompts/my_custom.pt

# 2. 修改配置
prompt:
  template_file: "prompts/my_custom.pt"
```

### 方法2: 使用绝对路径

```yaml
prompt:
  template_file: "/absolute/path/to/my_custom.pt"
```

## 验证路径设置

运行验证脚本检查所有路径：

```bash
cd /path/to/ChatXFEL/XFELBench
./bin/verify_setup.sh
```

**检查项**:
- ✅ [TEST 10] Checking prompt template accessibility

## 快速参考

| 配置项 | 基础目录 | 示例 |
|--------|---------|------|
| `prompt.template_file` | **ChatXFEL/** | `"prompts/naive.pt"` |
| `--questions` (命令行) | **XFELBench/** | `problem_sets/xfel_qa_basic.json` |
| `outputs/` | **XFELBench/** | 自动生成 |
| `evaluations/` | **XFELBench/** | 自动生成 |
| `configs/generated/` | **XFELBench/** | 自动生成 |

## 总结

记住关键点：
1. **Prompt路径相对于ChatXFEL根目录** ⭐
2. **其他所有路径相对于XFELBench根目录**
3. **从XFELBench目录运行所有命令**
4. **使用相对路径，不用绝对路径**

---

**文档更新日期**: 2025-01-02
