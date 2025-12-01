# ChatXFEL项目优化 - 三周开发计划

## 项目概述

**目标**: 优化ChatXFEL的检索过程，提高回答质量

**核心任务**（按优先级）:
1. 添加Rewrite和Rerank模块（高级RAG技术）
2. 子库路由：先检索摘要，再检索对应的全文
3. 实现元数据过滤，支持按关键词过滤文献
4. 混合检索：同时使用dense和sparse向量

**环境信息**:
- 大模型：qwen3:30b-a3b-instruct-2507-q8_0 (http://10.15.102.186:9000)
- 向量模型：BGE-M3 (支持dense+sparse双向量)
- 向量数据库：Milvus 2.5.22 (10.19.48.181:19530)
- 开发框架：LangChain

---

## 第一周：环境搭建 + Rewrite & Rerank增强

### Checkpoint 1.1: 环境配置（Day 1-2）

#### 任务清单
- [ ] 配置Milvus 2.5.22数据库连接
  - 数据库地址：10.19.48.181:19530
  - 用户名：cs286_2025_groupX (X=5或8)
  - 密码：GroupX
  - 数据库名：cs286_2025_groupX
  - Attu可视化界面：10.19.48.181:30411

- [ ] 部署/连接Qwen3大模型
  - 访问地址：http://10.15.102.186:9000
  - 模型名：qwen3:30b-a3b-instruct-2507-q8_0
  - 通过ollama访问

- [ ] 部署BGE-M3向量化模型
  - 模型名：bge-m3:latest
  - 特性：同时生成dense和sparse向量
  - 通过ollama访问

- [ ] 测试Reranker模型
  - 选项1：BGE-Reranker-v2-m3
  - 选项2：Qwen3-Reranker系列
  - 推荐使用BGE-Reranker-v2-m3（与现有代码兼容）

- [ ] 验证LangChain环境
  - 检查依赖包版本
  - 测试现有代码（chatxfel_app.py, rag.py）
  - 确保与新版本LangChain兼容

#### 验收标准
- 成功连接所有服务
- 能够调用模型进行简单的问答测试
- 现有代码能正常运行

---

### Checkpoint 1.2: Query Rewrite模块（Day 3-5）

#### 实现策略

**策略1：Query扩展**
```python
# 使用LLM扩展query，添加同义词和专业术语
prompt = """
请对以下XFEL领域的问题进行扩展，添加相关的同义词和专业术语：
问题：{original_query}
扩展后的问题：
"""
```

**策略2：Query分解**
```python
# 将复杂问题拆分成多个子问题
prompt = """
请将以下复杂问题分解为2-3个更简单的子问题：
问题：{original_query}
子问题：
1. 
2. 
3. 
"""
```

**策略3：回译增强（HyDE）**
```python
# 让LLM先生成假设性答案，用答案进行检索
prompt = """
请对以下问题生成一个假设性的答案：
问题：{original_query}
假设答案：
"""
```

#### 任务清单
- [ ] 在`rag.py`中添加`query_rewrite()`函数
- [ ] 实现至少2种rewrite策略
- [ ] 设计A/B测试：对比原始query vs. rewritten query
- [ ] 测试10个问题，记录检索结果差异
- [ ] 集成到现有RAG pipeline

#### 代码示例
```python
def query_rewrite(query: str, llm, strategy: str = 'expand') -> str:
    """
    重写用户查询
    
    Args:
        query: 原始查询
        llm: 大语言模型
        strategy: 'expand', 'decompose', 'hyde'
    
    Returns:
        重写后的查询
    """
    if strategy == 'expand':
        # 实现query扩展
        pass
    elif strategy == 'decompose':
        # 实现query分解
        pass
    elif strategy == 'hyde':
        # 实现HyDE
        pass
    return rewritten_query
```

#### 验收标准
- 完成至少2种rewrite策略
- 有对比实验数据
- 能够提升检索相关性

---

### Checkpoint 1.3: Rerank优化（Day 6-7）

#### 任务清单
- [ ] 分析现有rerank代码（`chatxfel_app.py`中已有）
- [ ] 优化参数：
  - `top_k`：初检数量（建议10-20）
  - `top_n`：rerank后保留数量（建议5-8）
- [ ] 实现two-stage reranking：
  - Stage 1: 快速粗排（使用简单模型或规则）
  - Stage 2: 精准精排（使用BGE-Reranker）
- [ ] 性能测试：记录rerank前后的相关性得分

#### 代码优化点
```python
# 现有代码中的rerank
compressor = get_rerank_model(top_n=n_recall)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever_obj.as_retriever(search_kwargs=search_kwargs)
)

# 优化建议：增加two-stage
def two_stage_rerank(docs, query, top_k=20, top_n=6):
    # Stage 1: 粗排（快速过滤）
    stage1_docs = coarse_rank(docs, query, top_k=top_k)
    
    # Stage 2: 精排（精准排序）
    stage2_docs = fine_rank(stage1_docs, query, top_n=top_n)
    
    return stage2_docs
```

#### 验收标准
- Rerank速度提升或效果提升
- 有详细的参数调优记录

---

## 第二周：混合检索 + 子库路由

### Checkpoint 2.1: 混合检索实现（Day 8-10）

#### 技术原理
BGE-M3模型可以同时生成：
- **Dense向量**：捕捉语义相似性
- **Sparse向量**：捕捉关键词匹配

混合检索公式：`score = α * dense_score + (1-α) * sparse_score`

#### 任务清单

**Step 1: 修改向量化代码**
- [ ] 更新`vectorize_bibs.py`
- [ ] 确保同时存储dense和sparse向量
- [ ] 验证现有数据是否已有双向量

```python
# 检查现有collection schema
from pymilvus import Collection, connections

connections.connect(**connection_args)
collection = Collection(name="your_collection")
print(collection.schema)  # 检查是否有sparse_vector字段
```

**Step 2: 创建混合检索collection**
- [ ] 在Milvus中创建新的collection
- [ ] Schema包含：
  - dense_vector (FLOAT_VECTOR, dim=1024)
  - sparse_vector (SPARSE_FLOAT_VECTOR)
  - 元数据字段（title, doi, journal, year等）

```python
# 参考vectorize_bibs.py中的create_bge_collection_by_connection()
fields = [
    FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR),
    # ... 其他字段
]
```

**Step 3: 实现混合检索**
- [ ] 在`rag.py`中添加`hybrid_search()`函数
- [ ] 实现加权策略（dense权重可调）
- [ ] 集成到retriever

```python
def hybrid_search(query, collection, embedding, dense_weight=0.7, top_k=10):
    """
    混合检索：dense + sparse
    
    Args:
        query: 查询文本
        collection: Milvus collection
        embedding: BGE-M3模型
        dense_weight: dense向量权重 (0-1)
        top_k: 返回结果数量
    
    Returns:
        检索结果
    """
    # 生成query的双向量
    query_vectors = embedding.encode_queries([query])
    dense_vec = query_vectors['dense'][0]
    sparse_vec = query_vectors['sparse'][0]
    
    # Milvus混合检索
    search_params = {
        "data": [[dense_vec], [sparse_vec]],
        "anns_field": ["dense_vector", "sparse_vector"],
        "param": [
            {"metric_type": "IP", "params": {"nprobe": 10}},
            {"metric_type": "IP", "params": {}}
        ],
        "limit": top_k,
        "weights": [dense_weight, 1-dense_weight]
    }
    
    results = collection.hybrid_search(**search_params)
    return results
```

**Step 4: 对比实验**
- [ ] 设计对比实验：
  - 纯dense检索
  - 纯sparse检索
  - 混合检索（不同权重）
- [ ] 测试20个问题
- [ ] 记录结果并分析

#### 验收标准
- 成功实现混合检索
- 有对比实验数据
- 找到最优的dense_weight参数

---

### Checkpoint 2.2: 子库路由系统（Day 11-14）

#### 系统架构

```
User Query
    ↓
[Abstract Collection] ← 第一步：检索相关论文摘要
    ↓ (获取DOI/Title)
[Fulltext Collection]  ← 第二步：检索对应的全文chunks
    ↓
Generate Answer
```

#### 任务清单

**Phase 1: 数据准备（Day 11）**
- [ ] 分析MongoDB中的论文数据结构
- [ ] 提取所有论文的摘要（abstract字段）
- [ ] 提取所有论文的全文chunks（已有的split结果）

```python
# 从MongoDB提取摘要
def extract_abstracts(mongo_collection):
    """提取所有论文摘要"""
    docs = mongo_collection.find(
        filter={'abstract': {'$ne': ''}},
        projection={'title': 1, 'doi': 1, 'abstract': 1, 'year': 1, 'journal': 1}
    )
    return list(docs)
```

**Phase 2: 创建摘要库（Day 11-12）**
- [ ] 创建abstract_collection
- [ ] Schema设计：
  ```python
  fields = [
      FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=1000),
      FieldSchema(name='doi', dtype=DataType.VARCHAR, max_length=1000, is_primary_key=True),
      FieldSchema(name='abstract', dtype=DataType.VARCHAR, max_length=10000),
      FieldSchema(name='year', dtype=DataType.INT16),
      FieldSchema(name='journal', dtype=DataType.VARCHAR, max_length=500),
      FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=1024),
      FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR),
  ]
  ```
- [ ] 向量化所有摘要并插入

**Phase 3: 创建全文库（Day 12）**
- [ ] 创建fulltext_collection（可能已存在）
- [ ] 确保每个chunk都关联到DOI/Title
- [ ] 添加索引以支持快速过滤

**Phase 4: 实现路由逻辑（Day 13-14）**
- [ ] 在`rag.py`中添加`route_retrieval()`函数

```python
def route_retrieval(query, abstract_collection, fulltext_collection, 
                   embedding, top_papers=5, top_chunks=10):
    """
    两阶段检索：先摘要后全文
    
    Args:
        query: 用户查询
        abstract_collection: 摘要库
        fulltext_collection: 全文库
        embedding: 向量化模型
        top_papers: 从摘要库检索的论文数
        top_chunks: 从每篇论文检索的chunk数
    
    Returns:
        最相关的文本chunks
    """
    # Step 1: 在摘要库中检索
    relevant_papers = hybrid_search(
        query=query,
        collection=abstract_collection,
        embedding=embedding,
        top_k=top_papers
    )
    
    # Step 2: 获取相关论文的DOI列表
    dois = [paper['doi'] for paper in relevant_papers]
    
    # Step 3: 在全文库中过滤检索
    # 只在这些DOI对应的chunks中搜索
    filter_expr = f"doi in {dois}"
    fulltext_results = hybrid_search(
        query=query,
        collection=fulltext_collection,
        embedding=embedding,
        top_k=top_chunks,
        filter=filter_expr
    )
    
    return fulltext_results
```

- [ ] 集成到主pipeline
- [ ] 添加fallback机制：如果摘要库未找到，直接搜全文

**Phase 5: 测试与优化（Day 14）**
- [ ] 对比实验：
  - 直接全文检索
  - 子库路由检索
- [ ] 调优参数：
  - top_papers（建议3-5）
  - top_chunks（建议每篇论文2-3个chunks）
- [ ] 分析检索速度和准确性

#### 验收标准
- 成功实现两阶段检索
- 检索精度有提升
- 检索速度可接受（建议<3秒）

---

## 第三周：元数据过滤 + Deep Research Agent（加分项）+ 评估优化

### Checkpoint 3.1: 元数据过滤系统（Day 15-16）

#### 功能需求
支持用户通过以下维度过滤文献：
1. **年份范围**（已有，需优化）
2. **期刊名称**
3. **关键词**（标题或摘要中包含）
4. **研究机构/装置**（facility字段）

#### 任务清单

**Task 1: 扩展Milvus Schema（Day 15）**
- [ ] 检查现有schema，确认所有需要的元数据字段
- [ ] 如需添加新字段（如keywords），更新schema
- [ ] 可能需要重新向量化部分数据

```python
# 添加keywords字段
schema.add_field(
    field_name='keywords',
    datatype=DataType.VARCHAR,
    max_length=500
)
```

**Task 2: 实现动态过滤（Day 15）**
- [ ] 在`rag.py`中添加`build_filter_expression()`函数

```python
def build_filter_expression(filters: dict) -> str:
    """
    构建Milvus过滤表达式
    
    Args:
        filters: {
            'year_range': (2018, 2024),
            'journals': ['Nature', 'Science'],
            'keywords': ['SFX', 'crystallography'],
            'facility': 'LCLS'
        }
    
    Returns:
        Milvus filter expression
    """
    expressions = []
    
    if 'year_range' in filters:
        start, end = filters['year_range']
        expressions.append(f"{start} <= year <= {end}")
    
    if 'journals' in filters:
        journals = filters['journals']
        journal_expr = " or ".join([f'journal == "{j}"' for j in journals])
        expressions.append(f"({journal_expr})")
    
    if 'keywords' in filters:
        keywords = filters['keywords']
        # 注意：Milvus的字符串匹配语法
        keyword_expr = " or ".join([f'title like "%{kw}%"' for kw in keywords])
        expressions.append(f"({keyword_expr})")
    
    if 'facility' in filters:
        expressions.append(f'facility == "{filters["facility"]}"')
    
    return " and ".join(expressions)
```

**Task 3: 更新UI（Day 16）**
- [ ] 在`chatxfel_app.py`中添加过滤选项
- [ ] 优化现有的`filter_year`功能
- [ ] 添加新的过滤控件

```python
# 在sidebar中添加
with st.sidebar:
    # 年份过滤（已有，保留）
    filter_year = st.checkbox('Filter by year', value=True)
    if filter_year:
        year_start = st.selectbox('Start year', ...)
        year_end = st.selectbox('End year', ...)
    
    # 期刊过滤（新增）
    filter_journal = st.checkbox('Filter by journal', value=False)
    if filter_journal:
        journals = st.multiselect(
            'Select journals',
            options=['Nature', 'Science', 'Physical Review', ...]
        )
    
    # 关键词过滤（新增）
    filter_keywords = st.checkbox('Filter by keywords', value=False)
    if filter_keywords:
        keywords_input = st.text_input(
            'Keywords (comma separated)',
            placeholder='SFX, crystallography, XFEL'
        )
        keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
    
    # 装置过滤（新增）
    filter_facility = st.checkbox('Filter by facility', value=False)
    if filter_facility:
        facility = st.selectbox(
            'Select facility',
            options=['LCLS', 'EuXFEL', 'SACLA', 'PAL-XFEL', ...]
        )
```

**Task 4: 集成到检索流程（Day 16）**
- [ ] 将过滤条件传递给retriever
- [ ] 测试各种过滤组合
- [ ] 确保过滤不影响检索速度

#### 验收标准
- 支持至少3种过滤维度
- UI交互流畅
- 过滤功能正确

---

### Checkpoint 3.1+: Deep Research Agent（Day 17-18，加分项）

> **重要说明**：此功能为**加分项**，应在确保核心功能（Rewrite、Rerank、混合检索、子库路由）都完成后再实施。如果时间紧张，可以实现简化版本或跳过此部分。

#### 功能目标

实现一个基于ReAct的Agent系统，能够：
1. **自动分解问题**：将复杂问题拆解成子问题
2. **多轮迭代检索**：根据中间结果决定下一步检索策略
3. **综合信息**：整合多个来源的信息生成深度报告
4. **展示思维过程**：让用户看到Agent的推理过程

**使用场景示例**：
- 用户问："总结XFEL在蛋白质结构解析中的应用和最新进展"
- Agent会：
  - Step 1: 先检索"XFEL protein structure"的基础知识
  - Step 2: 检索"SFX crystallography recent advances"
  - Step 3: 检索具体的应用案例
  - Step 4: 综合所有信息生成综述报告

#### ReAct框架原理

```
循环直到完成：
    Thought: 我现在需要了解什么？
    Action: 使用哪个工具？(search/retrieve/summarize)
    Observation: 工具返回了什么结果？
    [如果信息充分] → Final Answer
    [如果信息不足] → 继续循环
```

#### 任务清单

**Phase 1: 设计Agent Tools（Day 17上午）**

定义Agent可以使用的工具：

- [ ] **Tool 1: search_papers**
  ```python
  def search_papers(query: str, filters: dict = None) -> List[str]:
      """
      搜索相关论文标题和摘要
      
      Args:
          query: 搜索查询
          filters: 过滤条件（年份、期刊等）
      
      Returns:
          论文列表（标题+摘要片段）
      """
      # 使用abstract_collection检索
      pass
  ```

- [ ] **Tool 2: retrieve_details**
  ```python
  def retrieve_details(paper_titles: List[str], aspect: str) -> str:
      """
      获取特定论文的详细内容
      
      Args:
          paper_titles: 论文标题列表
          aspect: 关注的方面（如"methods", "results", "applications"）
      
      Returns:
          相关的详细内容
      """
      # 使用fulltext_collection检索特定论文的chunks
      pass
  ```

- [ ] **Tool 3: summarize_findings**
  ```python
  def summarize_findings(documents: List[str]) -> str:
      """
      总结当前已检索到的文献
      
      Args:
          documents: 文献内容列表
      
      Returns:
          摘要文本
      """
      # 使用LLM总结
      pass
  ```

**Phase 2: 实现ReAct Agent（Day 17下午）**

- [ ] 使用LangChain的ReAct框架

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# 定义ReAct提示词
REACT_PROMPT = """You are a research assistant specialized in XFEL (X-ray Free Electron Laser) literature.
Your goal is to thoroughly research the user's question by iteratively searching and analyzing papers.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: think about what information you need
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to answer
Final Answer: the final comprehensive answer to the original question

IMPORTANT GUIDELINES:
1. Break down complex questions into sub-questions
2. Search for 2-3 different aspects of the question
3. After each search, analyze if you have enough information
4. Aim for 3-5 iterations before giving final answer
5. Cite specific papers in your final answer

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

def create_research_agent(llm, tools):
    """创建ReAct研究Agent"""
    
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 显示思考过程
        max_iterations=6,  # 最多6轮迭代
        max_execution_time=120,  # 最多2分钟
        handle_parsing_errors=True
    )
    
    return agent_executor
```

- [ ] 集成tools到Agent

```python
# 创建工具列表
tools = [
    Tool(
        name="search_papers",
        func=search_papers,
        description="Useful for finding relevant papers. Input should be a search query string."
    ),
    Tool(
        name="retrieve_details",
        func=lambda x: retrieve_details(x.split(','), aspect="methods"),
        description="Useful for getting detailed content from specific papers. Input should be comma-separated paper titles."
    ),
    Tool(
        name="summarize_findings",
        func=summarize_findings,
        description="Useful for summarizing current research findings. Input should be 'summarize'."
    )
]

# 创建agent
research_agent = create_research_agent(llm, tools)
```

**Phase 3: 实现多轮对话和状态管理（Day 18上午）**

- [ ] 添加对话历史记录
- [ ] 实现中间结果缓存

```python
class ResearchSession:
    """管理一次Deep Research会话"""
    
    def __init__(self, agent_executor):
        self.agent = agent_executor
        self.history = []
        self.intermediate_results = {}
    
    def research(self, question: str) -> dict:
        """
        执行深度研究
        
        Returns:
            {
                'answer': 最终答案,
                'reasoning_steps': 推理步骤列表,
                'papers_used': 使用的论文列表,
                'iterations': 迭代次数
            }
        """
        # 记录推理过程
        reasoning_steps = []
        
        # 执行Agent
        result = self.agent.invoke(
            {"input": question},
            callbacks=[ReasoningCallback(reasoning_steps)]
        )
        
        # 提取使用的论文
        papers_used = self._extract_papers(reasoning_steps)
        
        return {
            'answer': result['output'],
            'reasoning_steps': reasoning_steps,
            'papers_used': papers_used,
            'iterations': len(reasoning_steps)
        }
    
    def _extract_papers(self, steps):
        """从推理步骤中提取引用的论文"""
        papers = []
        for step in steps:
            if 'Observation' in step:
                # 解析出论文标题
                pass
        return papers
```

**Phase 4: UI集成（Day 18下午）**

- [ ] 在Streamlit中添加"Deep Research"模式

```python
# 在chatxfel_app.py中添加
with st.sidebar:
    research_mode = st.radio(
        "Mode",
        options=["Quick Answer", "Deep Research"],
        help="Quick Answer: 单次检索\nDeep Research: 多轮迭代研究（较慢）"
    )
    
    if research_mode == "Deep Research":
        max_iterations = st.slider("Max iterations", 3, 8, 5)
        show_reasoning = st.checkbox("Show reasoning process", value=True)

# 在主流程中
if research_mode == "Deep Research":
    with st.spinner("🔬 Conducting deep research..."):
        session = ResearchSession(research_agent)
        result = session.research(question)
        
        # 显示推理过程
        if show_reasoning:
            with st.expander("🧠 Reasoning Process"):
                for i, step in enumerate(result['reasoning_steps']):
                    st.markdown(f"**Step {i+1}**")
                    st.text(step['thought'])
                    st.info(f"Action: {step['action']}")
                    st.success(f"Observation: {step['observation'][:200]}...")
        
        # 显示最终答案
        st.markdown("### 📊 Research Report")
        st.markdown(result['answer'])
        
        # 显示引用的论文
        with st.expander("📚 Papers Referenced"):
            for paper in result['papers_used']:
                st.markdown(f"- {paper}")
```

#### 简化方案（如果时间紧张）

如果时间不够，可以实现**最简版本**：

**核心功能**：
1. ✅ 只实现2个工具：search + summarize
2. ✅ 固定3轮迭代（不需要复杂的停止条件）
3. ✅ 在UI中添加一个"Deep Research"按钮

**代码示例**（简化版）：
```python
def simple_deep_research(question: str, llm, retriever) -> dict:
    """
    简化的Deep Research：固定3轮迭代
    """
    results = []
    
    # Round 1: 初步检索
    thought_1 = f"First, I need to understand the basics of: {question}"
    docs_1 = retriever.get_relevant_documents(question)
    summary_1 = llm.invoke(f"Summarize these papers: {docs_1[:3]}")
    results.append({
        'round': 1,
        'thought': thought_1,
        'action': 'search',
        'docs': docs_1[:3],
        'summary': summary_1
    })
    
    # Round 2: 深入特定方面
    thought_2 = "Now I need more specific information about applications"
    refined_query = f"{question} applications methods"
    docs_2 = retriever.get_relevant_documents(refined_query)
    summary_2 = llm.invoke(f"Summarize these papers: {docs_2[:3]}")
    results.append({
        'round': 2,
        'thought': thought_2,
        'action': 'search_specific',
        'docs': docs_2[:3],
        'summary': summary_2
    })
    
    # Round 3: 综合答案
    thought_3 = "Now I can synthesize all information"
    final_answer = llm.invoke(f"""Based on the following research:
    Round 1: {summary_1}
    Round 2: {summary_2}
    
    Please provide a comprehensive answer to: {question}
    """)
    results.append({
        'round': 3,
        'thought': thought_3,
        'action': 'synthesize',
        'answer': final_answer
    })
    
    return {
        'answer': final_answer,
        'steps': results
    }
```

#### 测试用例

设计3个测试问题（从简单到复杂）：

1. **简单问题**：
   - "What is serial femtosecond crystallography?"
   - 预期：2-3轮迭代即可

2. **中等问题**：
   - "Compare the data processing pipelines used at LCLS and EuXFEL"
   - 预期：4-5轮迭代

3. **复杂问题**：
   - "Summarize the evolution of XFEL technology from 2010 to 2024, focusing on improvements in pulse duration, repetition rate, and scientific applications"
   - 预期：5-6轮迭代

#### 验收标准

**必须完成**（简化版）：
- [ ] 实现固定3轮迭代的simple_deep_research
- [ ] 在UI中添加"Deep Research"模式
- [ ] 能够展示推理过程
- [ ] 测试至少1个问题

**加分完成**（完整版）：
- [ ] 实现完整的ReAct Agent
- [ ] 支持动态迭代次数
- [ ] 有详细的reasoning展示
- [ ] 测试3个不同复杂度的问题

#### 时间管理建议

**策略A（保守）**：
- Day 17: 如果前面进度正常，开始实现简化版
- Day 18: 完善和测试
- 如果时间不够，**放弃此功能**，专注核心功能

**策略B（激进）**：
- Day 17: 实现完整ReAct Agent
- Day 18: UI集成和测试
- 可能需要牺牲部分评估时间

**推荐采用策略A**，确保核心功能稳定。

---

### Checkpoint 3.2: 系统集成与测试（Day 19）

#### 完整Pipeline

```
User Query
    ↓
[Query Rewrite] ← 查询改写（扩展/分解）
    ↓
[Metadata Filter] ← 应用用户设定的过滤条件
    ↓
[Abstract Retrieval] ← 在摘要库中检索（混合检索）
    ↓
[Route to Fulltext] ← 路由到全文库
    ↓
[Fulltext Retrieval] ← 在全文chunks中检索（混合检索）
    ↓
[Rerank] ← Two-stage重排序
    ↓
[Generate Answer] ← LLM生成答案
    ↓
Response to User
```

#### 任务清单

**Day 19: 模块整合与测试**
- [ ] 创建新的主函数`advanced_rag_pipeline()`
- [ ] 集成所有模块：
  - query_rewrite
  - build_filter_expression
  - route_retrieval
  - hybrid_search
  - two_stage_rerank
- [ ] 添加错误处理和日志

```python
def advanced_rag_pipeline(query: str, 
                         llm, 
                         embedding,
                         abstract_collection,
                         fulltext_collection,
                         filters: dict = None,
                         use_rewrite: bool = True,
                         use_routing: bool = True) -> dict:
    """
    高级RAG pipeline
    
    Returns:
        {
            'answer': str,
            'context': List[Document],
            'metadata': {
                'rewritten_query': str,
                'papers_found': int,
                'retrieval_time': float
            }
        }
    """
    import time
    start_time = time.time()
    
    # Step 1: Query Rewrite
    if use_rewrite:
        rewritten_query = query_rewrite(query, llm, strategy='expand')
    else:
        rewritten_query = query
    
    # Step 2: Build Filter
    filter_expr = build_filter_expression(filters) if filters else None
    
    # Step 3: Retrieval
    if use_routing:
        # 两阶段检索
        docs = route_retrieval(
            query=rewritten_query,
            abstract_collection=abstract_collection,
            fulltext_collection=fulltext_collection,
            embedding=embedding,
            filter=filter_expr
        )
    else:
        # 直接全文检索
        docs = hybrid_search(
            query=rewritten_query,
            collection=fulltext_collection,
            embedding=embedding,
            filter=filter_expr
        )
    
    # Step 4: Rerank
    ranked_docs = two_stage_rerank(docs, rewritten_query, top_n=6)
    
    # Step 5: Generate
    answer = generate_answer(query, ranked_docs, llm)
    
    retrieval_time = time.time() - start_time
    
    return {
        'answer': answer,
        'context': ranked_docs,
        'metadata': {
            'rewritten_query': rewritten_query,
            'papers_found': len(ranked_docs),
            'retrieval_time': retrieval_time
        }
    }
```

- [ ] 单元测试：测试每个模块
- [ ] 集成测试：测试完整pipeline
- [ ] 性能测试：
  - 响应时间（目标<5秒）
  - 并发能力
  - 资源占用
- [ ] 压力测试：连续100次查询

#### 验收标准
- 所有模块正常工作
- Pipeline稳定运行
- 有完整的测试报告

---

### Checkpoint 3.3: 评估与报告（Day 20-21）

#### 评估任务1：文献库对比实验

**目标**：验证在大文献库中的检索一致性

**实验设计**：
- [ ] 构建测试集A：精选100篇高质量论文
- [ ] 构建测试集B：A + 900篇其他论文（共1000篇）
- [ ] 设计10个标准测试问题：
  ```
  1. What is serial femtosecond crystallography?
  2. How does XFEL compare to synchrotron radiation?
  3. What are the main data processing challenges in SFX?
  4. Describe the pump-probe technique in XFEL experiments.
  5. What is the typical pulse duration of XFEL?
  6. How to prepare samples for SPI experiments?
  7. What are the advantages of EuXFEL over LCLS?
  8. Explain the concept of hit-finding in XFEL data.
  9. What software tools are used for XFEL data analysis?
  10. What are recent developments in XFEL technology?
  ```

**评估指标**：
- [ ] Top-5文献重叠率：`overlap = len(set(A_docs) & set(B_docs)) / 5`
- [ ] 文献排序相关性：Kendall's τ
- [ ] 答案BLEU得分（如果A和B的答案应该相似）

**实验步骤**：
```python
def evaluate_consistency(questions, collection_A, collection_B, pipeline):
    """评估在不同大小文献库中的一致性"""
    results = []
    
    for q in questions:
        # 在A中检索
        docs_A = pipeline(q, collection=collection_A)
        
        # 在B中检索
        docs_B = pipeline(q, collection=collection_B)
        
        # 计算重叠率
        overlap = calculate_overlap(docs_A, docs_B, top_k=5)
        
        results.append({
            'question': q,
            'overlap_rate': overlap,
            'docs_A': docs_A,
            'docs_B': docs_B
        })
    
    return results
```

---

#### 评估任务2：回答质量评估

**目标**：评估回答的准确性和参考文献的相关性

**测试集**：
- [ ] 准备20个测试问题（覆盖不同难度）
  - 简单事实性问题（5个）
  - 中等复杂度问题（10个）
  - 复杂综合性问题（5个）

**评估维度**：
1. **答案准确性**（人工评分，1-5分）
   - 5分：完全准确，详细完整
   - 4分：基本准确，有少量遗漏
   - 3分：部分准确，有错误
   - 2分：大部分错误
   - 1分：完全错误

2. **参考文献相关性**（人工评分，1-5分）
   - 5分：所有文献高度相关
   - 4分：多数文献相关
   - 3分：部分文献相关
   - 2分：少数文献相关
   - 1分：文献不相关

3. **响应时间**（自动记录）
   - 目标：<5秒

**对比实验**：
- [ ] Baseline（优化前）：现有系统
- [ ] System-1：+ Rewrite + Rerank
- [ ] System-2：+ Hybrid Search
- [ ] System-3：+ Routing
- [ ] System-4（完整版）：所有优化

**评估代码**：
```python
def evaluate_qa_quality(questions, systems):
    """评估问答质量"""
    results = {sys_name: [] for sys_name in systems.keys()}
    
    for q in questions:
        for sys_name, sys_pipeline in systems.items():
            start_time = time.time()
            
            response = sys_pipeline(q)
            
            response_time = time.time() - start_time
            
            # 记录结果（人工评分后填入）
            results[sys_name].append({
                'question': q,
                'answer': response['answer'],
                'sources': response['context'],
                'response_time': response_time,
                'accuracy_score': None,  # 待人工评分
                'relevance_score': None  # 待人工评分
            })
    
    return results
```

---

#### 撰写项目报告（Day 21）

**报告结构**：

```markdown
# ChatXFEL系统优化报告

## 1. 项目概述
- 1.1 背景与目标
- 1.2 优化任务
- 1.3 技术栈

## 2. 技术实现

### 2.1 Query Rewrite模块
- 实现的策略
- 代码示例
- 效果对比

### 2.2 Hybrid Search（混合检索）
- Dense + Sparse双向量
- 权重调优
- 性能提升

### 2.3 子库路由系统
- 架构设计
- 实现细节
- 检索加速效果

### 2.4 Rerank优化
- Two-stage策略
- 参数调优
- 相关性提升

### 2.5 元数据过滤
- 支持的过滤维度
- UI设计
- 使用示例

### 2.6 Deep Research Agent（加分项）
- ReAct框架实现
- 工具设计
- 多轮迭代策略
- 思维过程展示

## 3. 实验结果

### 3.1 一致性测试
- 实验设计
- 数据统计（表格+图表）
- 结果分析

### 3.2 质量评估
- 评分统计
- 系统对比（柱状图）
- Case study（展示2-3个典型案例）

### 3.3 性能测试
- 响应时间对比
- 资源占用
- 并发能力

### 3.4 Deep Research Agent评估（如已实现）
- 迭代次数分析
- 信息覆盖度
- 答案深度对比（vs. Quick Answer模式）
- Case study展示

## 4. 问题与解决方案

### 4.1 遇到的主要问题
- 问题1: Milvus 2.5兼容性
  - 解决方案: ...
  
- 问题2: 向量化速度慢
  - 解决方案: ...

### 4.2 待改进之处
- 功能层面
- 性能层面
- 用户体验

## 5. 总结与展望

### 5.1 完成情况
- 已完成的功能
- 达成的目标

### 5.2 未来工作
- Short-term（3个月内）
- Long-term（6个月以上）

## 6. 附录
- 代码仓库链接
- 数据集说明
- 完整测试结果
```

**配图要求**：
- [ ] 系统架构图（使用draw.io或mermaid）
- [ ] Pipeline流程图
- [ ] 实验结果对比图（柱状图、折线图）
- [ ] UI截图

**数据统计表**：
- [ ] 各系统评分对比表
- [ ] 响应时间统计表
- [ ] 资源占用表

#### 验收标准
- 完成两项评估实验
- 有详细的数据和分析
- 提交完整的项目报告（PDF，15-20页）

---

## 关键里程碑总结

| 周次 | 关键产出 | 完成标志 |
|------|---------|---------|
| Week 1 | Rewrite + Rerank | 能够改写query并重排序结果 |
| Week 2 | 混合检索 + 子库路由 | 实现双向量检索和两阶段检索 |
| Week 3 | 元数据过滤 + Deep Research Agent（加分项）+ 完整评估 | 功能完备，有评估报告 |

**核心功能优先级**：
1. 🔴 **必须完成**：Rewrite, Rerank, 混合检索, 子库路由（Week 1-2）
2. 🟡 **重要功能**：元数据过滤（Week 3前期）
3. 🟢 **加分项**：Deep Research Agent（Week 3中期，时间允许时）
4. 🔵 **必须完成**：评估与报告（Week 3后期）

---

## 每日工作流程建议

### 每天开始（9:00-9:30）
- [ ] 检查GPU资源和服务状态
- [ ] 回顾昨天的进展
- [ ] 明确今天的目标

### 开发时段（9:30-12:00, 14:00-18:00）
- [ ] 集中开发
- [ ] 每2小时commit一次代码
- [ ] 记录实验结果到笔记

### 每天结束（18:00-18:30）
- [ ] 总结今天的工作
- [ ] 更新进度表
- [ ] 规划明天任务

### 每周五下午（16:00-18:00）
- [ ] 团队会议（如有）
- [ ] 演示本周成果
- [ ] 讨论下周计划

---

## Deep Research Agent实现决策指南

### 什么时候开始实现Deep Research？

**判断标准**（满足以下条件再开始）：
- ✅ Week 1-2的所有核心功能都已完成
- ✅ 基础RAG pipeline运行稳定
- ✅ 至少完成了一轮初步测试
- ✅ 还剩至少2天时间（Day 17-18）

### 实现哪个版本？

**简化版（推荐）**：
- 时间：1.5天
- 复杂度：低
- 功能：固定3轮迭代 + 基础UI
- 价值：展示Agent思维即可

**完整版（挑战）**：
- 时间：2天
- 复杂度：中
- 功能：动态ReAct + 完整工具链
- 价值：接近OpenAI Deep Research

### 如果时间不够怎么办？

**Plan B选项**：
1. **只写设计文档**：在报告中详细描述如何实现，作为"未来工作"
2. **Mock演示**：用预设的推理步骤模拟Agent思考过程
3. **放弃此功能**：专注于核心功能的打磨和评估

### 建议的时间分配

```
Day 15-16: 元数据过滤（必须完成）
Day 17上午: 评估是否开始Deep Research
    ├─ 如果进度良好 → 开始实现简化版
    └─ 如果进度延迟 → 跳过，开始Day 19的工作
Day 17下午-18: Deep Research实现（如果开始）
Day 19: 系统集成与测试（必须完成）
Day 20-21: 评估与报告（必须完成）
```

---

## 风险管理

### 高风险项（需提前准备）

**风险1: Milvus 2.5兼容性问题**
- 影响：可能无法使用新特性
- 应对：准备降级到2.4的方案，使用只读账号参考

**风险2: GPU资源不足**
- 影响：向量化速度慢，可能拖延进度
- 应对：
  - 分批处理数据
  - 使用CPU模式（速度慢但可用）
  - 调整batch size

**风险3: LangChain API变化**
- 影响：现有代码可能不兼容
- 应对：
  - 查阅最新文档
  - 参考官方migration guide
  - 保留旧版本代码作为备份

### 中风险项

**风险4: Deep Research实现时间不足**
- 影响：加分项无法完成
- 应对：
  - 优先确保核心功能完成
  - 实现简化版本
  - 或只在报告中描述设计方案

**风险5: LLM的ReAct能力不足**
- 影响：Agent无法正确推理
- 应对：
  - 简化prompt设计
  - 增加示例（few-shot）
  - 降级为固定流程的multi-step检索

**风险6: 数据质量问题**
- 影响：检索效果不佳
- 应对：手动筛选高质量论文子集进行测试

**风险7: 评估标准不明确**
- 影响：难以量化改进效果
- 应对：提前与导师沟通评估方式

---

## 资源清单

### 开发环境
- GPU服务器：（待分配）
- Milvus数据库：10.19.48.181:19530
- 大模型服务：http://10.15.102.186:9000

### 文档与工具
- Milvus官方文档：https://milvus.io/docs
- LangChain文档：https://python.langchain.com/
- BGE-M3论文：https://arxiv.org/abs/2402.03216
- 项目代码：（已提供5个文件）

### 数据资源
- 论文集下载：（见项目文档）
- 测试问题集：（需自行准备）

---

## 附录：快速参考

### 常用命令

```bash
# 连接Milvus
from pymilvus import connections
connections.connect(
    alias="default",
    host="10.19.48.181",
    port="19530",
    user="cs286_2025_groupX",
    password="GroupX"
)

# 调用Ollama模型
curl http://10.15.102.186:9000/api/generate -d '{
  "model": "qwen3:30b-a3b-instruct-2507-q8_0",
  "prompt": "What is XFEL?",
  "stream": false
}'

# 查看Milvus collection
from pymilvus import Collection
collection = Collection("your_collection_name")
print(collection.num_entities)
print(collection.schema)
```

### 重要提醒

1. **代码版本控制**
   - 每天至少commit 2次
   - 重要功能完成后立即commit
   - 写清楚commit message

2. **实验记录**
   - 创建Excel/Markdown记录所有实验
   - 记录参数、结果、观察
   - 截图保存重要结果

3. **定期备份**
   - 代码push到Git
   - 实验数据定期备份
   - 重要文件多处保存

4. **时间管理**
   - 严格按照checkpoint进行
   - 遇到困难及时调整
   - 优先完成核心功能

---

## 联系方式与求助

遇到问题时的求助顺序：

1. **查阅文档**：官方文档和已有代码
2. **搜索引擎**：GitHub Issues、Stack Overflow
3. **团队讨论**：与队友讨论（如有）
4. **向导师求助**：准备好问题描述和已尝试的方案

---

**祝项目顺利！加油！** 🚀
