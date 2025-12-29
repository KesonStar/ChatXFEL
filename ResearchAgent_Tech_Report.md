# ChatXFEL `research_agent` 技术报告（简版）

## 1. 背景与目标
`research_agent` 是 ChatXFEL 的“Deep Research”模式核心，用于把用户的研究主题转化为**结构化文献综述**。相比“Basic RAG”一次检索一次生成的流程，它引入“需求澄清 → 知识点大纲 → 分点检索 → 综述写作”的分阶段管线，以提升检索精度与综述组织质量。

代码入口：`research_agent/__init__.py` 中的 `DeepResearchAgent`，在 Streamlit UI `chatxfel_app.py` 的 “Deep Research” 模式中被调用。

## 2. 总体架构与工作流

### 2.1 模块划分
```
research_agent/
├── __init__.py              # DeepResearchAgent：流程编排
├── clarification.py         # ClarificationModule：澄清问题生成（JSON）
├── knowledge_extractor.py   # KnowledgeExtractor：知识点大纲生成（JSON）
├── research_planner.py      # ResearchPlanner：分知识点并行检索 +（可选）rerank + 去重
└── review_generator.py      # ReviewGenerator：综述生成 + 参考文献格式化
```

### 2.2 端到端流程（与实现一致）
1. **Clarification（可选）**：根据用户问题生成 2–4 个澄清问题（同语言输出，JSON）。  
2. **Knowledge Extraction**：结合用户原始问题 + 澄清回答，生成 4–8 个知识点大纲（JSON），每个知识点包含 topic、检索关键词、重要度等。  
3. **Parallel Search**：对每个知识点构建 query（topic + keywords），并发调用 `retriever.invoke(query)` 拉取候选文献；可选用 `reranker.compress_documents(docs, query)` 重排，并做跨知识点去重。  
4. **Review Generation**：将知识点大纲 + 按知识点组织的检索结果拼接进 prompt，调用 LLM 生成 Markdown 综述；参考文献单独格式化输出。

> 说明：接口设计中包含 `year_filter` 参数（UI 可选年份筛选），但当前 `ResearchPlanner.search()` 未实际应用该过滤条件（属于“接口预留/待实现”）。

## 3. 核心数据结构与接口约定

### 3.1 澄清问题（`ClarificationModule.generate_questions` 输出）
文件：`research_agent/clarification.py`
```json
{
  "questions": [
    {"id": 1, "question": "…", "purpose": "scope/focus/depth/timerange"}
  ]
}
```
实现要点：
- prompt 强约束“只输出 JSON”；解析时做“去 code block / 正则抽取 JSON / 失败回退默认问题”。

### 3.2 知识点大纲（`KnowledgeExtractor.extract` 输出）
文件：`research_agent/knowledge_extractor.py`
```json
{
  "title": "…",
  "knowledge_points": [
    {
      "id": "KP1",
      "category": "Core Concepts/Technical Methods/…",
      "topic": "…",
      "search_keywords": ["…", "…"],
      "importance": "high/medium/low"
    }
  ],
  "search_strategy": "…"
}
```
实现要点：
- LLM 输出 JSON 后进行结构校验（必须包含 `title`、`knowledge_points`，且每个知识点必须包含 `id/category/topic/search_keywords`）。
- 解析失败时回退到“基础大纲”（从问题中截取词作为关键词）。
- 支持 `update_outline(outline, modifications)` 以整包替换 title/knowledge_points/search_strategy（UI 编辑 JSON 时可复用）。

### 3.3 检索结果（`ResearchPlanner.search` 输出）
文件：`research_agent/research_planner.py`
```python
{
  "KP1": [Document, Document, ...],
  "KP2": [Document, Document, ...],
}
```
实现要点：
- query 构造策略：`query = topic + ' ' + ' '.join(search_keywords)`（当前实现无子 query / 无复杂布尔表达式）。
- 并行检索：`ThreadPoolExecutor(max_workers=min(len(queries), 4))`。
- 候选裁剪：每个知识点最多保留 `top_k_initial`（默认 20）。
- 可选 rerank：期望 reranker 具备 `compress_documents(docs, query)`；最终每个知识点最多保留 `top_k_final`（默认 10）。
- 跨知识点去重：优先使用 DOI；否则使用 title+page；再否则用内容片段 hash。

### 3.4 综述文本与参考文献
文件：`research_agent/review_generator.py`
- `generate(outline, search_results) -> str`：一次性生成完整 Markdown 综述（如开头没有 `#` 自动补标题）。
- `generate_section_by_section(...) -> str`：按 category 分组逐段生成（对长文可能更稳，但当前 UI 默认使用 `generate`）。
- `format_references(search_results) -> str`：去重后输出 Markdown 列表，DOI 自动拼接 `https://doi.org/{doi}` 链接。

## 4. 关键实现细节（对齐当前代码）
1. **LLM 调用方式统一**：各阶段均通过 `llm.invoke(prompt_text)` 调用；并用 `utils.strip_thinking_tags()` 去除思考模型的 `<think>...</think>` 前缀，保证 UI 仅展示最终输出。  
2. **JSON 解析鲁棒性**：澄清与大纲阶段都包含“清理 markdown 包裹 + 正则抓取 JSON”的容错，并提供默认/回退输出以避免 UI 卡死。  
3. **上下游边界清晰**：上游输出严格约定为 JSON（结构化），下游检索与生成只消费结构化结果，便于后续替换 prompt 或策略。  

## 5. 与 Streamlit UI 的集成方式
文件：`chatxfel_app.py`
- 模式选择：侧边栏 `Research Mode` 中选择 `Deep Research` 后进入分阶段 UI（`ss.dr_stage` 状态机）。  
- 阶段映射：
  - `initial`：输入主题；可选 “Quick Mode（跳过澄清）”
  - `clarification`：回答澄清问题
  - `confirmation`：展示并允许编辑知识点大纲 JSON
  - `searching`：执行分点检索并生成综述与参考文献
  - `review`：展示综述、参考文献，以及按知识点展开的文献列表

## 6. 运行与依赖（面向开发/复现实验）
1. 安装依赖：`pip install -r requirements.txt`  
2. 启动 UI：`streamlit run chatxfel_app.py`（如环境未安装 Streamlit：`pip install streamlit`）  
3. 使用 “Deep Research” 模式需要：
   - 可用的 LLM（当前通过 LangChain 的 `ChatOllama` 调用）
   - 可用的 retriever（当前项目主要基于 Milvus 的向量检索/混合检索）
   - 可选 reranker（cross-encoder rerank，用于提升相关性）

## 7. 当前限制与改进建议（基于实现现状）
- `year_filter` 参数当前未在 `ResearchPlanner` 内落地（可在 query/metadata filter 层实现）。  
- query 生成策略较朴素（topic + keywords 直连），后续可加入：
  - 同义词扩展、布尔检索表达式、facility/方法名约束
  - 每个知识点多 query（主 query + 子 query）并合并去重
- `generate()` 将每篇文献内容截断到 1000 字符拼进 prompt，长文/多文献场景下易触发上下文长度限制；可优先切换 UI 使用 `generate_section_by_section()` 或引入“先摘要再写作”的两段式生成。
