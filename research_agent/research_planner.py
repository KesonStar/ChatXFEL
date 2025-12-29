#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Planner Module for Deep Research Agent

Handles document retrieval for each knowledge point, including:
- Query generation from knowledge points
- Parallel document retrieval
- Reranking and deduplication
"""

import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from langchain_classic.schema.document import Document


class ResearchPlanner:
    """
    Module for planning and executing document searches for knowledge points.
    """

    def __init__(self, retriever, reranker=None, top_k_initial: int = 20, top_k_final: int = 10):
        """
        Initialize the research planner.

        Args:
            retriever: Document retriever (supports invoke() method)
            reranker: Optional reranker for improving retrieval quality
            top_k_initial: Number of initial candidates per knowledge point
            top_k_final: Number of final documents per knowledge point after reranking
        """
        self.retriever = retriever
        self.reranker = reranker
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final

    def search(self, outline: dict, year_filter: Tuple[int, int] = None) -> dict:
        """
        Search documents for all knowledge points in the outline.

        Args:
            outline: Knowledge point outline from KnowledgeExtractor
            year_filter: Optional (start_year, end_year) tuple

        Returns:
            dict mapping knowledge point IDs to lists of retrieved documents
            {
                'KP1': [Document, Document, ...],
                'KP2': [Document, Document, ...],
                ...
            }
        """
        knowledge_points = outline.get('knowledge_points', [])

        if not knowledge_points:
            return {}

        # Generate queries for each knowledge point
        queries = self._generate_queries(knowledge_points)

        # Retrieve documents (in parallel or sequentially)
        results = self._parallel_search(queries)

        # Rerank results if reranker is available
        if self.reranker:
            results = self._rerank_results(results, queries)

        # Deduplicate across knowledge points
        results = self._deduplicate_results(results)

        return results

    def _generate_queries(self, knowledge_points: List[dict]) -> Dict[str, str]:
        """
        Generate search queries from knowledge points.

        Args:
            knowledge_points: List of knowledge point dicts

        Returns:
            dict mapping KP IDs to search queries
        """
        queries = {}

        for kp in knowledge_points:
            kp_id = kp.get('id', 'unknown')
            topic = kp.get('topic', '')
            keywords = kp.get('search_keywords', [])

            # Combine topic and keywords into a search query
            query_parts = [topic]
            query_parts.extend(keywords)

            # Create a coherent search query
            query = ' '.join(query_parts)
            queries[kp_id] = query

        return queries

    def _parallel_search(self, queries: Dict[str, str]) -> Dict[str, List[Document]]:
        """
        Execute searches in parallel for all knowledge points.

        Args:
            queries: dict mapping KP IDs to search queries

        Returns:
            dict mapping KP IDs to lists of documents
        """
        results = {}

        # Use ThreadPoolExecutor for parallel retrieval
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            # Submit all search tasks
            future_to_kp = {
                executor.submit(self._search_single, query): kp_id
                for kp_id, query in queries.items()
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_kp):
                kp_id = future_to_kp[future]
                try:
                    docs = future.result()
                    results[kp_id] = docs
                except Exception as e:
                    print(f"Search failed for {kp_id}: {e}")
                    results[kp_id] = []

        return results

    def _search_single(self, query: str) -> List[Document]:
        """
        Execute a single search query.

        Args:
            query: Search query string

        Returns:
            List of retrieved documents
        """
        try:
            # Use invoke method for retrieval
            docs = self.retriever.invoke(query)

            # Limit to top_k_initial
            if len(docs) > self.top_k_initial:
                docs = docs[:self.top_k_initial]

            return docs
        except Exception as e:
            print(f"Search error for query '{query[:50]}...': {e}")
            return []

    def _rerank_results(self, results: Dict[str, List[Document]],
                       queries: Dict[str, str]) -> Dict[str, List[Document]]:
        """
        Rerank search results using the cross-encoder reranker.

        Args:
            results: dict mapping KP IDs to document lists
            queries: dict mapping KP IDs to original queries

        Returns:
            Reranked results dict
        """
        reranked = {}

        for kp_id, docs in results.items():
            if not docs:
                reranked[kp_id] = []
                continue

            query = queries.get(kp_id, '')

            try:
                # Use reranker to score documents
                # Note: This assumes the reranker has a compress_documents method
                # For cross-encoder reranker from langchain
                reranked_docs = self.reranker.compress_documents(docs, query)

                # Take top_k_final
                if len(reranked_docs) > self.top_k_final:
                    reranked_docs = reranked_docs[:self.top_k_final]

                reranked[kp_id] = list(reranked_docs)
            except Exception as e:
                print(f"Reranking failed for {kp_id}: {e}")
                # Fall back to original results
                reranked[kp_id] = docs[:self.top_k_final]

        return reranked

    def _deduplicate_results(self, results: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
        """
        Deduplicate documents across knowledge points.

        Documents that appear in multiple KP results are kept only in their
        most relevant KP (first occurrence in results order).

        Args:
            results: dict mapping KP IDs to document lists

        Returns:
            Deduplicated results dict
        """
        seen_docs = {}  # Maps doc identifier to KP ID where it first appeared
        deduped = {}

        for kp_id, docs in results.items():
            deduped_docs = []

            for doc in docs:
                # Create a unique identifier for the document
                doc_id = self._get_doc_identifier(doc)

                if doc_id not in seen_docs:
                    # First occurrence, keep it
                    seen_docs[doc_id] = kp_id
                    deduped_docs.append(doc)
                else:
                    # Already seen, add reference to metadata
                    # but don't include in this KP's results
                    pass

            deduped[kp_id] = deduped_docs

        return deduped

    def _get_doc_identifier(self, doc: Document) -> str:
        """
        Get a unique identifier for a document.

        Uses DOI if available, otherwise title + page.

        Args:
            doc: Document object

        Returns:
            Unique identifier string
        """
        metadata = doc.metadata

        # Try DOI first
        doi = metadata.get('doi', '')
        if doi:
            return f"doi:{doi}"

        # Fall back to title + page
        title = metadata.get('title', '')
        page = metadata.get('page', '')
        if title:
            return f"title:{title}:page:{page}"

        # Last resort: use content hash
        return f"content:{hash(doc.page_content[:100])}"

    def get_all_documents(self, results: Dict[str, List[Document]]) -> List[Document]:
        """
        Get all unique documents from results.

        Args:
            results: dict mapping KP IDs to document lists

        Returns:
            Flat list of all unique documents
        """
        all_docs = []
        seen = set()

        for docs in results.values():
            for doc in docs:
                doc_id = self._get_doc_identifier(doc)
                if doc_id not in seen:
                    seen.add(doc_id)
                    all_docs.append(doc)

        return all_docs

    def get_document_count(self, results: Dict[str, List[Document]]) -> Dict[str, int]:
        """
        Get document count per knowledge point.

        Args:
            results: Search results dict

        Returns:
            dict mapping KP IDs to document counts
        """
        return {kp_id: len(docs) for kp_id, docs in results.items()}
