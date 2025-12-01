# ChatXFEL Enhancement Implementation Plan

## Project Overview

This document outlines the implementation roadmap for three core enhancements to the ChatXFEL system:

1. **Query Rewrite**: Optimize user queries using conversation history to improve retrieval accuracy
2. **Chat History Management**: Enable multi-turn conversation with context awareness
3. **Agent-based Deep Research (ReAct Agent)**: Implement an intelligent research assistant using the ReAct framework

---

## Phase 1: Query Rewrite

### Objective
Address vague or context-dependent user queries by rewriting them into standalone, precise search queries using conversation history.

### Key Tasks
- [ ] Design query rewriting strategy and prompt templates
- [ ] Implement Query Rewriter module (`query_rewriter.py`)
- [ ] Integrate into existing RAG pipeline (modify `rag.py` and `chatxfel_app.py`)
- [ ] Add UI toggle for enabling/disabling query rewrite
- [ ] Test and evaluate rewriting effectiveness

### Key Files
- New: `query_rewriter.py`, `prompts/query_rewrite.pt`
- Modified: `rag.py`, `chatxfel_app.py`

### Expected Outcomes
- 15-25% improvement in retrieval accuracy for ambiguous queries
- Better handling of pronouns and references in multi-turn conversations

---

## Phase 2: Chat History Management

### Objective
Fully leverage conversation history to enable context-aware, fluent multi-turn dialogues.

### Key Tasks
- [ ] Analyze and optimize current history storage mechanism
- [ ] Enhance history processing functions in `rag.py`
- [ ] Design prompt templates that incorporate chat history
- [ ] Implement history length management and truncation strategy
- [ ] Add history control options in UI
- [ ] Test multi-turn conversation and reference resolution

### Key Files
- New: `prompts/chat_with_history.pt`
- Modified: `rag.py`, `chatxfel_app.py`

### Expected Outcomes
- Seamless multi-turn conversation support
- Ability to resolve references like "it", "that method", etc.
- Significant improvement in contextual understanding

---

## Phase 3: Agent-based Deep Research (ReAct Agent)

### Objective
Build an intelligent research assistant that uses multi-step reasoning and tool execution to answer complex research questions.

### ReAct Framework Overview
ReAct = Reasoning + Action + Observation
- **Reasoning**: Analyze current state and decide next action
- **Action**: Execute tool calls (search, analyze, synthesize)
- **Observation**: Process tool outputs
- Iterate until reaching final answer

### Key Tasks
- [ ] Design ReAct framework and define tool set
- [ ] Implement ReAct Agent core engine (`react_agent.py`)
- [ ] Design ReAct prompt template
- [ ] Implement iteration control and reasoning trace tracking
- [ ] Integrate into main system with Agent mode option
- [ ] Visualize reasoning process in UI
- [ ] Test and optimize Agent performance

### Agent Toolset
- `search_papers`: Retrieve relevant papers from vector database
- `analyze_paper`: Deep analysis of specific paper content
- `cross_reference`: Cross-reference multiple papers
- `synthesize`: Combine findings into coherent answer

### Key Files
- New: `react_agent.py`, `prompts/react_agent.pt`
- Modified: `chatxfel_app.py`

### Expected Outcomes
- Handle complex research questions requiring multi-step reasoning
- Provide explainable reasoning process with intermediate steps
- Automatically synthesize information from multiple papers

---

## Phase 4: System Integration & Testing

### Key Tasks
- [ ] Create unified configuration management (`config.py`)
- [ ] Implement mode switching (Basic RAG / RAG+History+Rewrite / Agent Mode)
- [ ] End-to-end functional testing
- [ ] Performance benchmarking and optimization
- [ ] Update documentation (README.md, AGENTS.md, user guide)
- [ ] Code review and quality assurance

### Testing Focus
- Single-turn dialogue (Basic RAG)
- Multi-turn dialogue (with History & Rewrite)
- Complex research questions (Agent Mode)
- Performance and resource usage evaluation

---

## Implementation Priority

### High Priority (P0) - Core Features
1. Query Rewrite basic implementation
2. Chat History enhancement
3. ReAct Agent framework setup

### Medium Priority (P1) - Feature Optimization
4. Query Rewrite fine-tuning
5. ReAct Agent toolset expansion
6. UI enhancement and visualization

---


## Technical Requirements

### New Dependencies
- `langchain-core` and `langchain-community` (already present)
- Optional: `graphviz` (for Agent visualization)

### Configuration Management
Support three operation modes:
- Basic RAG
- RAG with Rewrite & History
- Agent Mode

---
