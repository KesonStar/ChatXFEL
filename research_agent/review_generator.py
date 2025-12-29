#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review Generator Module for Deep Research Agent

Generates structured literature reviews based on knowledge outlines
and retrieved documents.
"""

import sys
sys.path.append('..')
import utils

from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.document import Document


REVIEW_GENERATION_TEMPLATE = """You are an expert scientific writer specializing in X-ray Free-Electron Laser (XFEL) research.

Your task is to write a comprehensive, structured literature review based on the provided knowledge outline and retrieved scientific literature.

Guidelines:
1. **Structure**: Follow the knowledge outline to organize the review
2. **Citations**: Use [Author, Year] format for citations (e.g., [Smith et al., 2023])
3. **Synthesis**: Don't just summarize individual papers - synthesize findings across multiple sources
4. **Critical Analysis**: Compare and contrast different approaches, identify agreements and disagreements
5. **Technical Accuracy**: Include specific technical details (parameters, methods, results) from the papers
6. **Academic Style**: Maintain formal academic writing style
7. **Research Gaps**: Identify gaps and future research directions at the end

Important:
- Do NOT include a reference list at the end (references will be shown separately)
- Focus on synthesizing information across multiple papers
- Highlight key findings, trends, and developments
- Be specific with technical details when available

Knowledge Outline:
Title: {title}

Knowledge Points:
{knowledge_points_text}

Retrieved Literature by Knowledge Point:
{documents_text}

Write a comprehensive literature review following the outline structure:"""


SECTION_GENERATION_TEMPLATE = """You are an expert scientific writer specializing in X-ray Free-Electron Laser (XFEL) research.

Write a detailed section for a literature review based on the following knowledge point and retrieved documents.

Knowledge Point:
- Category: {category}
- Topic: {topic}
- Importance: {importance}

Retrieved Documents:
{documents_text}

Guidelines:
1. Synthesize findings across multiple sources
2. Use [Author, Year] citation format
3. Include specific technical details
4. Compare and contrast different approaches
5. Be comprehensive but focused on the topic
6. Use formal academic writing style

Write the section content (without section header):"""


class ReviewGenerator:
    """
    Module for generating structured literature reviews.
    """

    def __init__(self, llm):
        """
        Initialize the review generator.

        Args:
            llm: Language model for review generation
        """
        self.llm = llm
        self.main_prompt = PromptTemplate(
            template=REVIEW_GENERATION_TEMPLATE,
            input_variables=["title", "knowledge_points_text", "documents_text"]
        )
        self.section_prompt = PromptTemplate(
            template=SECTION_GENERATION_TEMPLATE,
            input_variables=["category", "topic", "importance", "documents_text"]
        )

    def generate(self, outline: dict, search_results: Dict[str, List[Document]]) -> str:
        """
        Generate a complete literature review.

        Args:
            outline: Knowledge point outline
            search_results: Retrieved documents organized by knowledge point

        Returns:
            Markdown-formatted literature review
        """
        title = outline.get('title', 'Literature Review')
        knowledge_points = outline.get('knowledge_points', [])

        # Format knowledge points as text
        kp_text = self._format_knowledge_points(knowledge_points)

        # Format documents as text
        docs_text = self._format_documents(knowledge_points, search_results)

        # Generate review using main prompt
        formatted_prompt = self.main_prompt.format(
            title=title,
            knowledge_points_text=kp_text,
            documents_text=docs_text
        )

        result = self.llm.invoke(formatted_prompt)

        # Extract content
        if hasattr(result, 'content'):
            review_text = result.content
        else:
            review_text = str(result)

        # Strip thinking tags if present
        review_text = utils.strip_thinking_tags(review_text)

        # Add title if not present
        if not review_text.strip().startswith('#'):
            review_text = f"# {title}\n\n{review_text}"

        return review_text

    def generate_section_by_section(self, outline: dict,
                                    search_results: Dict[str, List[Document]]) -> str:
        """
        Generate review section by section (for better quality on long reviews).

        Args:
            outline: Knowledge point outline
            search_results: Retrieved documents organized by knowledge point

        Returns:
            Markdown-formatted literature review
        """
        title = outline.get('title', 'Literature Review')
        knowledge_points = outline.get('knowledge_points', [])

        # Group knowledge points by category
        categories = self._group_by_category(knowledge_points)

        # Build review sections
        sections = []
        sections.append(f"# {title}\n")

        # Generate introduction
        intro = self._generate_introduction(title, knowledge_points)
        sections.append(f"## 1. Introduction\n\n{intro}\n")

        # Generate each category section
        section_num = 2
        for category, kps in categories.items():
            sections.append(f"## {section_num}. {category}\n")

            for kp in kps:
                kp_id = kp.get('id')
                topic = kp.get('topic', '')
                docs = search_results.get(kp_id, [])

                if docs:
                    section_content = self._generate_section(kp, docs)
                    sections.append(f"### {section_num}.{kps.index(kp)+1} {topic}\n\n{section_content}\n")

            section_num += 1

        # Generate research gaps and future directions
        gaps = self._generate_gaps_section(knowledge_points, search_results)
        sections.append(f"## {section_num}. Research Gaps and Future Directions\n\n{gaps}\n")
        section_num += 1

        # Generate conclusion
        conclusion = self._generate_conclusion(title, knowledge_points)
        sections.append(f"## {section_num}. Conclusion\n\n{conclusion}\n")

        return "\n".join(sections)

    def _format_knowledge_points(self, knowledge_points: List[dict]) -> str:
        """
        Format knowledge points as text for the prompt.

        Args:
            knowledge_points: List of knowledge point dicts

        Returns:
            Formatted text string
        """
        lines = []
        for kp in knowledge_points:
            lines.append(f"- {kp.get('id')}: [{kp.get('category')}] {kp.get('topic')} (Importance: {kp.get('importance', 'medium')})")

        return "\n".join(lines)

    def _format_documents(self, knowledge_points: List[dict],
                         search_results: Dict[str, List[Document]]) -> str:
        """
        Format documents as text for the prompt.

        Args:
            knowledge_points: List of knowledge point dicts
            search_results: Retrieved documents by KP ID

        Returns:
            Formatted text string
        """
        sections = []

        for kp in knowledge_points:
            kp_id = kp.get('id')
            topic = kp.get('topic', '')
            docs = search_results.get(kp_id, [])

            sections.append(f"\n### {kp_id}: {topic}")

            if not docs:
                sections.append("No relevant documents found for this topic.")
                continue

            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                title = metadata.get('title', 'Unknown')
                year = metadata.get('year', 'N/A')
                journal = metadata.get('journal', '')
                doi = metadata.get('doi', '')

                # Format document info
                doc_info = f"\n**Document {i}**: {title}"
                if year != 'N/A':
                    doc_info += f" ({year})"
                if journal:
                    doc_info += f" - {journal}"
                if doi:
                    doc_info += f" [DOI: {doi}]"

                sections.append(doc_info)
                sections.append(f"Content: {doc.page_content[:1000]}...")  # Truncate for prompt

        return "\n".join(sections)

    def _group_by_category(self, knowledge_points: List[dict]) -> Dict[str, List[dict]]:
        """
        Group knowledge points by category.

        Args:
            knowledge_points: List of knowledge point dicts

        Returns:
            dict mapping categories to lists of knowledge points
        """
        categories = {}
        for kp in knowledge_points:
            category = kp.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append(kp)

        return categories

    def _generate_section(self, kp: dict, docs: List[Document]) -> str:
        """
        Generate a single section of the review.

        Args:
            kp: Knowledge point dict
            docs: List of documents for this knowledge point

        Returns:
            Section content string
        """
        # Format documents for this section
        docs_text = self._format_docs_for_section(docs)

        formatted_prompt = self.section_prompt.format(
            category=kp.get('category', ''),
            topic=kp.get('topic', ''),
            importance=kp.get('importance', 'medium'),
            documents_text=docs_text
        )

        result = self.llm.invoke(formatted_prompt)

        if hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)

        return utils.strip_thinking_tags(content)

    def _format_docs_for_section(self, docs: List[Document]) -> str:
        """
        Format documents for section generation prompt.

        Args:
            docs: List of documents

        Returns:
            Formatted text
        """
        lines = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            title = metadata.get('title', 'Unknown')
            year = metadata.get('year', 'N/A')
            authors = self._extract_authors(metadata)

            lines.append(f"\n**Paper {i}**: {title}")
            if authors:
                lines.append(f"Authors: {authors}")
            lines.append(f"Year: {year}")
            lines.append(f"Content excerpt:\n{doc.page_content}")

        return "\n".join(lines)

    def _extract_authors(self, metadata: dict) -> str:
        """
        Extract author information from metadata.

        Args:
            metadata: Document metadata dict

        Returns:
            Author string or empty string
        """
        # Try various metadata keys for authors
        for key in ['authors', 'author', 'Authors', 'Author']:
            if key in metadata and metadata[key]:
                return str(metadata[key])
        return ""

    def _generate_introduction(self, title: str, knowledge_points: List[dict]) -> str:
        """
        Generate introduction section.

        Args:
            title: Review title
            knowledge_points: List of knowledge points

        Returns:
            Introduction text
        """
        topics = [kp.get('topic', '') for kp in knowledge_points[:3]]

        intro_prompt = f"""Write a brief introduction (2-3 paragraphs) for a literature review titled "{title}".

The review will cover the following main topics:
{', '.join(topics)}

The introduction should:
1. Provide background context
2. State the scope and objectives of the review
3. Briefly outline the structure

Write the introduction:"""

        result = self.llm.invoke(intro_prompt)

        if hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)

        return utils.strip_thinking_tags(content)

    def _generate_gaps_section(self, knowledge_points: List[dict],
                              search_results: Dict[str, List[Document]]) -> str:
        """
        Generate research gaps and future directions section.

        Args:
            knowledge_points: List of knowledge points
            search_results: Search results

        Returns:
            Section content
        """
        # Collect all high-importance topics
        high_priority = [kp.get('topic') for kp in knowledge_points
                        if kp.get('importance') == 'high']

        gaps_prompt = f"""Based on a literature review covering the following topics:
{', '.join([kp.get('topic', '') for kp in knowledge_points])}

Write a "Research Gaps and Future Directions" section (2-3 paragraphs) that:
1. Identifies potential gaps in the current research
2. Suggests promising directions for future investigation
3. Highlights areas needing more attention

Write the section:"""

        result = self.llm.invoke(gaps_prompt)

        if hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)

        return utils.strip_thinking_tags(content)

    def _generate_conclusion(self, title: str, knowledge_points: List[dict]) -> str:
        """
        Generate conclusion section.

        Args:
            title: Review title
            knowledge_points: List of knowledge points

        Returns:
            Conclusion text
        """
        conclusion_prompt = f"""Write a conclusion (1-2 paragraphs) for a literature review titled "{title}".

The review covered the following topics:
{', '.join([kp.get('topic', '') for kp in knowledge_points])}

The conclusion should:
1. Summarize key findings
2. Emphasize the significance of the research area
3. Provide a forward-looking statement

Write the conclusion:"""

        result = self.llm.invoke(conclusion_prompt)

        if hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)

        return utils.strip_thinking_tags(content)

    def format_references(self, search_results: Dict[str, List[Document]]) -> str:
        """
        Format all references from search results.

        Args:
            search_results: Search results dict

        Returns:
            Formatted reference list
        """
        seen = set()
        references = []

        for kp_id, docs in search_results.items():
            for doc in docs:
                metadata = doc.metadata
                doi = metadata.get('doi', '')
                title = metadata.get('title', 'Unknown')

                # Use DOI or title as identifier
                ref_id = doi if doi else title
                if ref_id in seen:
                    continue
                seen.add(ref_id)

                # Format reference
                year = metadata.get('year', 'N/A')
                journal = metadata.get('journal', '')

                ref = f"- {title}"
                if year != 'N/A':
                    ref += f" ({year})"
                if journal:
                    ref += f". {journal}"
                if doi:
                    ref += f". DOI: [{doi}](https://doi.org/{doi})"

                references.append(ref)

        return "\n".join(references)
