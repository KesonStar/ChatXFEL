# ChatXFEL Deep Research Agent 架构设计

## 文档信息
- **版本**: 1.0  
- **日期**: 2024-12-16
- **目标**: 实现简化版的Deep Research Agent，用于生成高质量的文献综述

---

## 1. 设计概述

### 1.1 核心理念
在生成最终literature review前，系统通过与用户的一轮对话澄清需求，并生成结构化的知识点大纲，基于此大纲进行精准的文献检索和综述生成。

### 1.2 设计优势
- **降低实现复杂度**: 相比完整的ReAct Agent更轻量
- **提高检索质量**: 通过需求澄清和结构化生成更精准的检索query
- **用户掌控感**: 用户可参与并调整研究方向
- **渐进式开发**: 可先实现简化版，后续扩展

### 1.3 潜在挑战
- 用户体验：额外的交互步骤可能显得繁琐
- 知识点提取质量：依赖LLM的理解能力
- 时间成本：增加了交互环节

---

## 2. 系统架构

### 2.1 模块结构
```
research_agent/
├── __init__.py
├── clarification.py         # 需求澄清模块
├── knowledge_extractor.py   # 知识点提取模块
├── research_planner.py      # 研究计划与检索模块
└── review_generator.py      # 文献综述生成模块
```

### 2.2 工作流程
```
用户原始问题
    ↓
【阶段1：需求澄清】
生成澄清问题 (2-4个精准问题)
    ↓
用户回答 ← → Agent追问 (可选，最多1轮)
    ↓
【阶段2：知识点提取】
整合用户原始问题 + 回答
    ↓
生成结构化知识点大纲
    ↓
用户确认/修改大纲 (可选)
    ↓
【阶段3：文献检索】
将知识点转为多个精准检索query
    ↓
并行检索相关文献 (每个知识点独立检索)
    ↓
Rerank + 去重 + 分组
    ↓
【阶段4：综述生成】
基于检索到的文献 + 知识点大纲
    ↓
生成结构化Literature Review
```

---

## 3. 核心模块设计

### 3.1 需求澄清模块 (clarification.py)

**功能职责**
- 分析用户原始问题
- 生成2-4个精准的澄清问题
- 帮助理解研究范围、重点、深度等需求

**输出格式**
```json
{
  "questions": [
    {
      "id": 1, 
      "question": "具体澄清问题",
      "purpose": "clarify scope/focus/depth"
    }
  ]
}
```

**设计要点**
- 澄清问题应涵盖：研究范围边界、关注的具体方面、时间范围、期望深度
- 简化版本固定只问一轮，不追问
- 输出标准化JSON格式便于后续处理

---

### 3.2 知识点提取模块 (knowledge_extractor.py)

**功能职责**
- 整合用户原始问题和澄清答案
- 生成结构化的知识点大纲
- 为每个知识点提取检索关键词

**输出格式**
```json
{
  "title": "文献综述标题",
  "knowledge_points": [
    {
      "id": "KP1",
      "category": "分类（如Core Concepts, Technical Methods等）",
      "topic": "知识点主题",
      "search_keywords": ["关键词1", "关键词2"],
      "importance": "high/medium/low"
    }
  ],
  "search_strategy": "检索策略简述"
}
```

**设计要点**
- 知识点应涵盖：核心概念/定义、关键研究问题、技术方面、研究趋势、对比分析
- 每个知识点配备精准的检索关键词
- 知识点之间应有逻辑层次关系

**输出示例**
```json
{
  "title": "Recent Advances in Serial Femtosecond Crystallography Data Processing",
  "knowledge_points": [
    {
      "id": "KP1",
      "category": "Core Concepts",
      "topic": "SFX data collection principles",
      "search_keywords": ["serial femtosecond crystallography", "SFX", "data collection"],
      "importance": "high"
    },
    {
      "id": "KP2",
      "category": "Technical Methods",
      "topic": "Hit-finding algorithms",
      "search_keywords": ["hit finding", "peak detection", "Bragg spots"],
      "importance": "high"
    }
  ],
  "search_strategy": "Search each knowledge point independently, prioritize papers from 2020-2024"
}
```

---

### 3.3 研究计划与检索模块 (research_planner.py)

**功能职责**
- 将知识点转换为精准的检索queries
- 并行检索所有知识点相关文献
- 对检索结果进行Rerank和去重
- 按知识点组织检索结果

**核心功能**
1. **Query生成**: 为每个知识点生成主query和子queries
2. **并行检索**: 同时检索所有知识点（每个知识点top_k=20获取候选集）
3. **Rerank**: 对每个知识点的候选文献重排序（降至top_k=10）
4. **去重**: 跨知识点去重，保留文献在最相关知识点下

**检索策略**
```
为每个知识点生成:
{
  'KP_ID': {
    'main_query': '主检索query',
    'sub_queries': ['子query1', '子query2'],
    'filters': {
      'year': [起始年, 结束年],
      'facility': ['设施名']  # 可选
    }
  }
}
```

**去重策略**
- 基于DOI或title识别重复文献
- 保留文献在最相关的知识点下
- 记录文献同时关联的其他知识点

---

### 3.4 综述生成模块 (review_generator.py)

**功能职责**
- 基于知识点大纲和检索到的文献生成结构化综述
- 整合分析各知识点下的文献
- 进行批判性分析和综合
- 识别研究空白和未来方向

**综述结构**
```markdown
# 文献综述标题

## 1. Introduction
研究背景、范围、核心问题

## 2. [知识点1类别]
### 2.1 [知识点1主题]
基于相关文献的分析...

### 2.2 [知识点2主题]
...

## 3. [知识点2类别]
...

## 4. Research Gaps and Future Directions
当前研究的局限性和未来发展方向

## 5. Conclusion
核心发现总结
```

**设计要点**
- 遵循知识点大纲的结构组织内容
- 使用[Author, Year]格式引用文献
- 对比和综合不同研究方法
- 识别研究趋势和空白
- 保持学术写作风格
- 包含文献中的具体技术细节
- **重要**: 综述正文中不包含参考文献列表（遵循系统prompt要求）

---

## 4. 用户界面流程

### 4.1 状态管理
系统通过以下状态控制研究流程：
- `research_stage`: 当前所处阶段（initial/clarification/confirmation/searching/review）
- `original_question`: 用户原始问题
- `clarification_questions`: 生成的澄清问题
- `clarifications`: 用户的澄清答案
- `knowledge_outline`: 提取的知识点大纲
- `search_results`: 检索到的文献（按知识点组织）
- `final_review`: 最终生成的综述

### 4.2 交互流程

**阶段1: Initial - 问题输入**
- 用户输入研究主题
- 系统生成澄清问题
- 进入Clarification阶段

**阶段2: Clarification - 需求澄清**
- 展示2-4个澄清问题
- 用户逐一回答
- 提交后进入Confirmation阶段

**阶段3: Confirmation - 大纲确认**
- 展示生成的知识点大纲（JSON格式）
- 用户可以：
  - 确认并开始检索
  - 编辑大纲（修改JSON）
- 确认后进入Searching阶段

**阶段4: Searching - 文献检索**
- 系统并行检索所有知识点
- 显示检索进度
- 完成后进入Review阶段

**阶段5: Review - 综述展示**
- 展示生成的文献综述（Markdown格式）
- 可展开查看各知识点下的参考文献列表
- 提供"开始新研究"按钮重置流程

### 4.3 模式选择
在侧边栏提供两种模式：
- **Basic RAG**: 传统的快速问答
- **Deep Research**: 深度文献综述（使用本架构）

---

## 5. 简化版本方案（MVP）

如果时间紧张，可以实现最小可行产品：

### 5.1 简化流程
```
用户输入问题
    ↓
直接生成知识点大纲（跳过澄清对话）
    ↓
用户确认/编辑大纲
    ↓
并行检索
    ↓
生成综述
```

### 5.2 实现差异
- 省略clarification模块
- knowledge_extractor直接从原始问题生成大纲
- 其他模块保持不变

### 5.3 优势
- 开发周期更短
- 用户体验更简洁
- 仍保留核心价值（结构化检索 + 综述生成）

---

## 6. 关键设计决策

### 6.1 为什么选择一轮澄清对话
- 平衡了用户体验和需求理解
- 避免过多交互导致的用户流失
- 简化实现复杂度

### 6.2 为什么使用知识点大纲
- 将复杂问题分解为可管理的子问题
- 支持并行检索提高效率
- 提供清晰的综述结构
- 用户可控可编辑

### 6.3 为什么并行检索
- 显著提升检索速度
- 每个知识点独立检索提高相关性
- 便于后续按主题组织文献

### 6.4 为什么分离Rerank和去重
- Rerank确保每个知识点下文献质量
- 去重避免综述中重复引用
- 分离处理提高灵活性

---

## 7. 与现有系统的集成

**本架构**可以：
- 复用现有的retriever、reranker、LLM配置
- 作为独立模块添加到现有系统
- 通过侧边栏模式切换使用

**集成要点**：
- 使用相同的Milvus连接和collection
- 使用相同的BGE-M3 embedding和BGE-Reranker-v2-m3
- 使用相同的Qwen3 LLM（通过Ollama）
- 新增独立的prompt文件和模块代码


---

## 9. 总结

### 9.1 设计特点
- **轻量化**: 相比完整ReAct Agent更简单实用
- **结构化**: 知识点大纲提供清晰的研究框架
- **可控性**: 用户可参与并调整研究方向
- **高效性**: 并行检索显著提升性能

### 9.2 核心价值
- 将复杂的文献综述任务分解为可管理的子任务
- 提供比传统RAG更全面深入的文献分析
- 通过结构化方法提高检索精度和综述质量

### 9.3 适用场景
- 需要全面了解某个研究主题
- 需要结构化的文献综述
- 需要对比分析不同研究方向
- 需要识别研究趋势和空白

---

**文档版本历史**
- v1.0 (2024-12-16): 初始架构设计版本