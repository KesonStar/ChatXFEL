#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatXFEL Deep Research Agent

A module for generating high-quality literature reviews through structured
knowledge extraction and parallel document retrieval.

Workflow:
1. Clarification: Generate and collect clarifying questions
2. Knowledge Extraction: Create structured knowledge point outline
3. Research Planning: Convert knowledge points to search queries and retrieve documents
4. Review Generation: Generate structured literature review

Usage:
    from research_agent import DeepResearchAgent

    agent = DeepResearchAgent(llm, retriever, reranker)

    # Stage 1: Get clarification questions
    questions = agent.generate_clarification_questions(user_question)

    # Stage 2: Generate knowledge outline (after user answers)
    outline = agent.extract_knowledge_points(user_question, user_answers)

    # Stage 3: Search documents
    results = agent.search_documents(outline)

    # Stage 4: Generate review
    review = agent.generate_review(outline, results)
"""

from .clarification import ClarificationModule
from .knowledge_extractor import KnowledgeExtractor
from .research_planner import ResearchPlanner
from .review_generator import ReviewGenerator

__all__ = [
    'ClarificationModule',
    'KnowledgeExtractor',
    'ResearchPlanner',
    'ReviewGenerator',
    'DeepResearchAgent'
]


class DeepResearchAgent:
    """
    Main agent class that orchestrates the deep research workflow.
    """

    def __init__(self, llm, retriever, reranker=None):
        """
        Initialize the Deep Research Agent.

        Args:
            llm: Language model for generation (ChatOllama)
            retriever: Document retriever (Milvus-based)
            reranker: Optional reranker for improving retrieval quality
        """
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker

        # Initialize modules
        self.clarification = ClarificationModule(llm)
        self.knowledge_extractor = KnowledgeExtractor(llm)
        self.research_planner = ResearchPlanner(retriever, reranker)
        self.review_generator = ReviewGenerator(llm)

    def generate_clarification_questions(self, question: str) -> dict:
        """
        Stage 1: Generate clarification questions for the user's research topic.

        Args:
            question: User's original research question

        Returns:
            dict with 'questions' key containing list of clarification questions
        """
        return self.clarification.generate_questions(question)

    def extract_knowledge_points(self, question: str, clarifications: dict) -> dict:
        """
        Stage 2: Extract structured knowledge points from user input.

        Args:
            question: User's original research question
            clarifications: User's answers to clarification questions

        Returns:
            dict containing knowledge point outline
        """
        return self.knowledge_extractor.extract(question, clarifications)

    def search_documents(self, outline: dict, year_filter: tuple = None) -> dict:
        """
        Stage 3: Search documents for each knowledge point.

        Args:
            outline: Knowledge point outline from Stage 2
            year_filter: Optional (start_year, end_year) tuple for filtering

        Returns:
            dict mapping knowledge point IDs to retrieved documents
        """
        return self.research_planner.search(outline, year_filter)

    def generate_review(self, outline: dict, search_results: dict) -> str:
        """
        Stage 4: Generate the final literature review.

        Args:
            outline: Knowledge point outline
            search_results: Retrieved documents organized by knowledge point

        Returns:
            Markdown-formatted literature review
        """
        return self.review_generator.generate(outline, search_results)

    def run_full_pipeline(self, question: str, clarifications: dict = None,
                         year_filter: tuple = None) -> dict:
        """
        Run the complete research pipeline (for simplified/MVP mode).

        Args:
            question: User's research question
            clarifications: Optional user clarifications
            year_filter: Optional year filter

        Returns:
            dict with 'outline', 'search_results', 'review' keys
        """
        # Extract knowledge points
        outline = self.extract_knowledge_points(question, clarifications or {})

        # Search documents
        search_results = self.search_documents(outline, year_filter)

        # Generate review
        review = self.generate_review(outline, search_results)

        return {
            'outline': outline,
            'search_results': search_results,
            'review': review
        }
