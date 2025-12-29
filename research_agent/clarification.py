#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clarification Module for Deep Research Agent

Generates clarification questions to better understand user's research needs.
"""

import json
import re
import sys
sys.path.append('..')
import utils

from langchain_core.prompts import PromptTemplate


CLARIFICATION_TEMPLATE = """You are a research assistant specializing in X-ray Free-Electron Laser (XFEL) and related scientific fields.

Your task is to analyze the user's research question and generate 2-4 precise clarification questions to better understand their research needs.

The clarification questions should cover:
1. **Research Scope**: What specific aspects or sub-topics should be included/excluded?
2. **Research Focus**: Which particular technologies, methods, or facilities are most relevant?
3. **Time Range**: What time period should the literature review cover?
4. **Depth Level**: How detailed should the review be (overview vs. in-depth technical analysis)?

Guidelines:
- Generate questions that will help create a more targeted and useful literature review
- Questions should be specific and answerable
- Avoid redundant or overly broad questions
- Questions should be in the same language as the user's input
- Return your response as valid JSON only

User's Research Question:
{question}

Response Format (JSON only, no markdown code blocks):
{{
  "questions": [
    {{
      "id": 1,
      "question": "Your clarification question here",
      "purpose": "scope/focus/depth/timerange"
    }},
    {{
      "id": 2,
      "question": "Your second clarification question here",
      "purpose": "scope/focus/depth/timerange"
    }}
  ]
}}

Generate 2-4 clarification questions:"""


class ClarificationModule:
    """
    Module for generating clarification questions to refine research scope.
    """

    def __init__(self, llm):
        """
        Initialize the clarification module.

        Args:
            llm: Language model for question generation
        """
        self.llm = llm
        self.prompt = PromptTemplate(
            template=CLARIFICATION_TEMPLATE,
            input_variables=["question"]
        )

    def generate_questions(self, question: str) -> dict:
        """
        Generate clarification questions for a research topic.

        Args:
            question: User's original research question

        Returns:
            dict with 'questions' key containing list of question dicts,
            each with 'id', 'question', and 'purpose' keys

        Example output:
            {
                "questions": [
                    {"id": 1, "question": "...", "purpose": "scope"},
                    {"id": 2, "question": "...", "purpose": "focus"}
                ]
            }
        """
        # Format prompt
        formatted_prompt = self.prompt.format(question=question)

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
            # Try to extract JSON from the response
            parsed = self._parse_json_response(response_text)
            return parsed
        except Exception as e:
            print(f"Failed to parse clarification questions: {e}")
            # Return default questions on failure
            return self._get_default_questions()

    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON from LLM response, handling various formats.

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

    def _get_default_questions(self) -> dict:
        """
        Return default clarification questions when parsing fails.

        Returns:
            dict with default questions
        """
        return {
            "questions": [
                {
                    "id": 1,
                    "question": "What specific aspects of this topic are you most interested in?",
                    "purpose": "scope"
                },
                {
                    "id": 2,
                    "question": "Are there any particular techniques or methods you want to focus on?",
                    "purpose": "focus"
                },
                {
                    "id": 3,
                    "question": "What time period should the literature review cover (e.g., last 5 years, all time)?",
                    "purpose": "timerange"
                }
            ]
        }
