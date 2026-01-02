# 导入路径修复总结

## 问题描述

在重组XFELBench文件结构后，遇到了模块导入错误：

```
ModuleNotFoundError: No module named 'rag'
ModuleNotFoundError: No module named 'scripts.generation'
```

## 根本原因

XFELBench中的脚本需要导入两类模块：

1. **ChatXFEL主项目模块** (`rag.py`, `utils.py`, `query_rewriter.py`)
   - 位置：`/ChatXFEL/` (主项目根目录)

2. **XFELBench内部模块** (`generate_configs.py`, 等)
   - 位置：`/ChatXFEL/XFELBench/scripts/`

重组后，脚本的相对路径发生变化，导致无法正确找到这些模块。

## 文件层级结构

```
ChatXFEL/                           # 主项目根目录
├── rag.py                          # 需要导入
├── utils.py                        # 需要导入
├── query_rewriter.py               # 需要导入
└── XFELBench/                      # 子项目根目录
    ├── scripts/
    │   ├── __init__.py            # ← 新增
    │   ├── evaluation/
    │   │   ├── __init__.py        # ← 新增
    │   │   ├── eval_generator.py  # 需要导入 rag, utils
    │   │   ├── llm_judge.py
    │   │   └── compare_results.py
    │   ├── generation/
    │   │   ├── __init__.py        # ← 新增
    │   │   └── generate_configs.py
    │   └── orchestration/
    │       ├── __init__.py        # ← 新增
    │       └── run_full_evaluation.py  # 需要导入 generate_configs
    └── ...
```

## 修复方案

### 1. 创建Python包结构

为所有 `scripts/` 子目录添加 `__init__.py`：

```bash
touch scripts/__init__.py
touch scripts/evaluation/__init__.py
touch scripts/generation/__init__.py
touch scripts/orchestration/__init__.py
```

### 2. 修复 `eval_generator.py` 的导入路径和Prompt路径

**文件位置**: `scripts/evaluation/eval_generator.py`

**需要导入**: `ChatXFEL/` 根目录的模块

**修复前**:
```python
# 错误：只向上2层到 scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**修复后**:
```python
# 正确：向上4层到 ChatXFEL/
# Current: ChatXFEL/XFELBench/scripts/evaluation/eval_generator.py
# Target:  ChatXFEL/
CHATXFEL_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(CHATXFEL_ROOT))
```

**路径计算**:
- `Path(__file__)` = `.../eval_generator.py`
- `.parent` = `.../evaluation/`
- `.parent.parent` = `.../scripts/`
- `.parent.parent.parent` = `.../XFELBench/`
- `.parent.parent.parent.parent` = `.../ChatXFEL/` ✅

**同时修复Prompt模板路径** ⭐:

配置文件中 `template_file: "prompts/naive.pt"` 是相对于 **ChatXFEL根目录** 的路径！

**修复前**:
```python
# 错误：相对于 scripts/ 目录
prompt_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    prompt_file
)
```

**修复后**:
```python
# 正确：相对于 ChatXFEL 根目录
if not os.path.isabs(prompt_file):
    prompt_file = os.path.join(str(CHATXFEL_ROOT), prompt_file)
```

现在会正确找到 `/ChatXFEL/prompts/naive.pt` ✅

### 3. 修复 `run_full_evaluation.py` 的导入路径

**文件位置**: `scripts/orchestration/run_full_evaluation.py`

**需要导入**: `scripts/generation/generate_configs.py`

**修复前**:
```python
from scripts.generation.generate_configs import ...
```

**修复后**:
```python
# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import from sibling package
from generation.generate_configs import ...
```

### 4. 修复 `generate_configs.py` 的输出路径

**问题**: 输出目录计算错误

**修复前**:
```python
# 错误：基于脚本所在目录
output_path = Path(__file__).parent / output_dir
```

**修复后**:
```python
# 正确：基于XFELBench根目录
xfelbench_root = Path(__file__).parent.parent.parent
output_path = xfelbench_root / output_dir
```

### 5. 修复 `run_full_evaluation.py` 的 base_dir

**问题**: 基础目录设置错误

**修复前**:
```python
# 错误：设置为 scripts/orchestration/
self.base_dir = Path(__file__).parent
```

**修复后**:
```python
# 正确：向上2层到 XFELBench/
self.base_dir = Path(__file__).parent.parent.parent
```

## 验证修复

运行验证脚本：

```bash
./bin/verify_setup.sh
```

**预期输出**:
```
[TEST 1] Testing config generator --list...
  ✅ PASSED
[TEST 2] Testing config generation...
  ✅ PASSED
[TEST 3] Testing run_full_evaluation --list-configs...
  ✅ PASSED
...
[TEST 9] Checking import paths to ChatXFEL...
  ✅ PASSED

✅ All tests passed! Setup is correct.
```

## 依赖问题说明

修复导入路径后，可能遇到依赖缺失错误：

```
ModuleNotFoundError: No module named 'langchain_community'
```

**这不是路径问题**，而是Python包未安装。需要：

```bash
pip install langchain-community langchain-classic
```

或参考主项目的 `requirements.txt`：

```bash
cd /path/to/ChatXFEL
pip install -r requirements.txt
```

## 路径计算速查表

| 文件 | 当前位置 | 需要访问 | 向上层数 | 代码 | 用途 |
|------|---------|---------|---------|------|------|
| `eval_generator.py` | `scripts/evaluation/` | `ChatXFEL/` | 4层 | `Path(__file__).parent.parent.parent.parent` | 导入rag模块 |
| `eval_generator.py` | `scripts/evaluation/` | `ChatXFEL/` | 4层 | `CHATXFEL_ROOT` (已定义) | Prompt路径 ⭐ |
| `run_full_evaluation.py` | `scripts/orchestration/` | `scripts/` | 1层 | `Path(__file__).parent.parent` | 导入generate_configs |
| `generate_configs.py` | `scripts/generation/` | `XFELBench/` | 2层 | `Path(__file__).parent.parent.parent` | 输出configs/ |
| `run_full_evaluation.py` (base_dir) | `scripts/orchestration/` | `XFELBench/` | 2层 | `Path(__file__).parent.parent.parent` | 工作目录 |

**重要说明**:
- ⭐ Prompt模板路径（`prompts/naive.pt`）在配置文件中是相对于 **ChatXFEL根目录** 的
- 在 `eval_generator.py` 中使用 `CHATXFEL_ROOT` 变量来解析prompt路径
- 所有相对路径都需要基于正确的根目录计算

## 最佳实践

### 1. 使用 `Path` 而非字符串操作

✅ **推荐**:
```python
from pathlib import Path
root = Path(__file__).parent.parent
```

❌ **不推荐**:
```python
import os
root = os.path.dirname(os.path.dirname(__file__))
```

### 2. 使用 `sys.path.insert(0, ...)` 而非 `append`

✅ **推荐**:
```python
sys.path.insert(0, str(CHATXFEL_ROOT))  # 优先级最高
```

❌ **不推荐**:
```python
sys.path.append(str(CHATXFEL_ROOT))  # 优先级低
```

### 3. 添加清晰的注释

```python
# Add ChatXFEL root directory to path
# Current: ChatXFEL/XFELBench/scripts/evaluation/eval_generator.py
# Target:  ChatXFEL/
CHATXFEL_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(CHATXFEL_ROOT))
```

### 4. 创建 `__init__.py` 标识Python包

确保所有目录都有 `__init__.py`，即使是空文件。

## 测试命令

### 测试配置生成

```bash
python scripts/generation/generate_configs.py --list
```

### 测试完整pipeline

```bash
python scripts/orchestration/run_full_evaluation.py --list-configs
```

### 测试路径计算

```bash
python -c "from pathlib import Path; print(Path('scripts/evaluation/eval_generator.py').parent.parent.parent.parent)"
```

## 故障排除

### 问题1: `ModuleNotFoundError: No module named 'rag'`

**原因**: 导入路径未正确指向ChatXFEL根目录

**解决**: 检查并修复 `CHATXFEL_ROOT` 计算

### 问题2: `ModuleNotFoundError: No module named 'scripts.generation'`

**原因**: 缺少 `__init__.py` 或 `sys.path` 设置错误

**解决**:
1. 确保所有目录有 `__init__.py`
2. 使用相对导入：`from generation.generate_configs import ...`

### 问题3: 输出目录创建在错误位置

**原因**: `base_dir` 或 `output_path` 计算错误

**解决**: 确保基础目录指向XFELBench根目录

## 总结

✅ **所有导入路径问题已修复**

关键修复点：
1. 添加 `__init__.py` 文件
2. 正确计算到ChatXFEL根目录的路径（4层）
3. 正确设置XFELBench base_dir（2-3层）
4. 使用 `Path` 对象处理路径
5. 使用 `sys.path.insert(0, ...)` 确保优先级

现在XFELBench可以正确导入所有需要的模块！

---

**修复日期**: 2025-01-02
**状态**: ✅ 完成并验证
