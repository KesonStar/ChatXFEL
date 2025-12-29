#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Extractor Module for Deep Research Agent

Extracts structured knowledge points from user's research question and clarifications.
"""

import json
import re
import sys
sys.path.append('..')
import utils

from langchain_core.prompts import PromptTemplate


KNOWLEDGE_EXTRACTION_TEMPLATE = """You are a research planning assistant specializing in X-ray Free-Electron Laser (XFEL) and related scientific fields.

Your task is to analyze the user's research question and their clarification answers, then generate a structured knowledge point outline for a literature review.

Guidelines for Knowledge Point Extraction:
1. **Core Concepts**: Fundamental definitions, principles, and theoretical background
2. **Technical Methods**: Specific techniques, algorithms, experimental methods
3. **Applications**: Use cases, experiments, real-world applications
4. **Comparative Analysis**: Comparison between different approaches, facilities, or methods
5. **Recent Advances**: Latest developments, trends, and future directions

Each knowledge point should:
- Have a clear, specific topic
- Include 2-4 search keywords optimized for scientific literature retrieval
- Be assigned an importance level (high/medium/low)
- Be logically connected to other knowledge points

Common XFEL Abbreviations to consider:
- XFEL: X-ray Free-Electron Laser
- SFX: Serial Femtosecond Crystallography
- SPI: Single Particle Imaging
- SASE: Self-Amplified Spontaneous Emission
- LCLS: Linac Coherent Light Source
- SACLA: SPring-8 Angstrom Compact Free Electron Laser
- EuXFEL: European XFEL
- SHINE: Shanghai High Repetition Rate XFEL and Extreme Light Facility

User's Original Question:
{question}

User's Clarification Answers:
{clarifications}

Response Format (JSON only, no markdown code blocks):
{{
  "title": "Proposed Literature Review Title",
  "knowledge_points": [
    {{
      "id": "KP1",
      "category": "Core Concepts/Technical Methods/Applications/Comparative Analysis/Recent Advances",
      "topic": "Specific topic description",
      "search_keywords": ["keyword1", "keyword2", "keyword3"],
      "importance": "high/medium/low"
    }}
  ],
  "search_strategy": "Brief description of overall search strategy"
}}

Generate 4-8 knowledge points that comprehensively cover the research topic:"""


class KnowledgeExtractor:
    """
    Module for extracting structured knowledge points from research questions.
    """

    def __init__(self, llm):
        """
        Initialize the knowledge extractor.

        Args:
            llm: Language model for knowledge extraction
        """
        self.llm = llm
        self.prompt = PromptTemplate(
            template=KNOWLEDGE_EXTRACTION_TEMPLATE,
            input_variables=["question", "clarifications"]
        )

    def extract(self, question: str, clarifications: dict = None) -> dict:
        """
        Extract knowledge points from user's question and clarifications.

        Args:
            question: User's original research question
            clarifications: dict mapping question IDs to user answers

        Returns:
            dict containing knowledge point outline with 'title', 'knowledge_points',
            and 'search_strategy' keys

        Example output:
            {
                "title": "Literature Review Title",
                "knowledge_points": [
                    {
                        "id": "KP1",
                        "category": "Core Concepts",
                        "topic": "...",
                        "search_keywords": ["..."],
                        "importance": "high"
                    }
                ],
                "search_strategy": "..."
            }
        """
        # Format clarifications as text
        clarifications_text = self._format_clarifications(clarifications)

        # Format prompt
        formatted_prompt = self.prompt.format(
            question=question,
            clarifications=clarifications_text
        )

        # Get LLM response
        result = self.llm.invoke(formatted_prompt)

        # Extract content
        if hasattr(result, 'content'):
            response_text = result.content
        else:
            response_text = str(result)

        # Strip thinking tags if present (for thinking models)
        response_text = utils.strip_thinking_tags(response_text)

        # Parse JSON response
        try:
            parsed = self._parse_json_response(response_text)
            # Validate structure
            self._validate_outline(parsed)
            return parsed
        except Exception as e:
            print(f"Failed to parse knowledge outline: {e}")
            # Return a basic outline on failure
            return self._generate_basic_outline(question)

    def _format_clarifications(self, clarifications: dict) -> str:
        """
        Format clarification answers as text.

        Args:
            clarifications: dict mapping question IDs to answers

        Returns:
            Formatted text string
        """
        if not clarifications:
            return "No additional clarifications provided."

        lines = []
        for q_id, answer in clarifications.items():
            lines.append(f"Q{q_id}: {answer}")

        return "\n".join(lines)

    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON from LLM response.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON as dict
        """
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {response_text[:200]}...")

    def _validate_outline(self, outline: dict) -> None:
        """
        Validate the structure of the knowledge outline.

        Args:
            outline: Parsed outline dict

        Raises:
            ValueError: If outline structure is invalid
        """
        if 'title' not in outline:
            raise ValueError("Outline missing 'title'")
        if 'knowledge_points' not in outline:
            raise ValueError("Outline missing 'knowledge_points'")
        if not isinstance(outline['knowledge_points'], list):
            raise ValueError("'knowledge_points' must be a list")
        if len(outline['knowledge_points']) == 0:
            raise ValueError("'knowledge_points' cannot be empty")

        for kp in outline['knowledge_points']:
            required_fields = ['id', 'category', 'topic', 'search_keywords']
            for field in required_fields:
                if field not in kp:
                    raise ValueError(f"Knowledge point missing '{field}'")

    def _generate_basic_outline(self, question: str) -> dict:
        """
        Generate a basic outline when parsing fails.

        Args:
            question: Original question

        Returns:
            Basic outline dict
        """
        # Extract key terms from question for search
        keywords = question.split()[:5]

        return {
            "title": f"Literature Review: {question[:50]}...",
            "knowledge_points": [
                {
                    "id": "KP1",
                    "category": "Core Concepts",
                    "topic": "Background and fundamentals",
                    "search_keywords": keywords[:3] if len(keywords) >= 3 else keywords,
                    "importance": "high"
                },
                {
                    "id": "KP2",
                    "category": "Technical Methods",
                    "topic": "Methods and techniques",
                    "search_keywords": keywords[:3] if len(keywords) >= 3 else keywords,
                    "importance": "high"
                },
                {
                    "id": "KP3",
                    "category": "Recent Advances",
                    "topic": "Recent developments",
                    "search_keywords": keywords[:3] if len(keywords) >= 3 else keywords,
                    "importance": "medium"
                }
            ],
            "search_strategy": "Search using main topic keywords across all knowledge points"
        }

    def update_outline(self, outline: dict, modifications: dict) -> dict:
        """
        Update an existing outline with user modifications.

        Args:
            outline: Original outline
            modifications: dict with updates (can modify title, add/remove/edit knowledge points)

        Returns:
            Updated outline
        """
        updated = outline.copy()

        if 'title' in modifications:
            updated['title'] = modifications['title']

        if 'knowledge_points' in modifications:
            # Replace entire knowledge points list
            updated['knowledge_points'] = modifications['knowledge_points']

        if 'search_strategy' in modifications:
            updated['search_strategy'] = modifications['search_strategy']

        return updated
