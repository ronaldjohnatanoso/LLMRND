"""LLM-based commitment extraction from text.

Extracts commitments with meta-roles from paragraphs using LLM.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from cog_memory.node import Node, Role

if TYPE_CHECKING:
    from collections.abc import Sequence


load_dotenv()


class LLMExtractor:
    """Extract commitments from text using LLM.

    Parses paragraphs into structured commitments with meta-role assignments.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.3,
        use_dummy: bool = False,
    ) -> None:
        """Initialize the LLM extractor.

        Args:
            model: Model name (uses OPENAI_MODEL env var if None)
            temperature: Sampling temperature for generation
            use_dummy: Use dummy extractor for testing (no API calls)
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.temperature = temperature
        self.use_dummy = use_dummy

        if not use_dummy:
            from openai import OpenAI

            self.client = OpenAI()

    def _build_extraction_prompt(self, text: str) -> str:
        """Build the prompt for commitment extraction.

        Args:
            text: Input text to extract commitments from

        Returns:
            Formatted prompt for the LLM
        """
        return f"""Extract commitments from the following text and assign each a meta-role.

Meta-roles:
- fact: Verifiable information
- observation: Noted information without verification
- goal: Target state or objective to achieve
- constraint: Limitation or restriction on actions
- decision: Chosen course of action
- conditional_dependency: Relationship between nodes (if X then Y)

For each commitment, provide:
1. text: The exact commitment text
2. role: One of the meta-roles above
3. confidence: Score from 0.0 to 1.0

Return as JSON array:
[
    {{"text": "...", "role": "fact", "confidence": 0.9}},
    {{"text": "...", "role": "goal", "confidence": 0.8}}
]

Text to analyze:
{text}

Response:"""

    def extract_commitments(self, text: str) -> list[Node]:
        """Extract commitments from text.

        Args:
            text: Input paragraph or text

        Returns:
            List of Node objects representing extracted commitments
        """
        if self.use_dummy:
            return self._dummy_extract(text)

        prompt = self._build_extraction_prompt(text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise commitment extractor. Always return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            result = response.choices[0].message.content
            if not result:
                return []

            data = json.loads(result)
            commitments = data.get("commitments", data)

            nodes = []
            for item in commitments:
                nodes.append(
                    Node(
                        text=item["text"],
                        role=Role(item["role"]),
                        confidence=item.get("confidence", 0.7),
                    )
                )

            return nodes

        except Exception as e:
            print(f"Error extracting commitments: {e}")
            return self._dummy_extract(text)

    def _dummy_extract(self, text: str) -> list[Node]:
        """Dummy extractor for testing (splits by sentences).

        Args:
            text: Input text

        Returns:
            List of basic Node objects
        """
        import re

        sentences = re.split(r"[.!?]+", text)
        nodes = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Simple heuristic for role assignment
            role = Role.FACT
            lower = sentence.lower()

            if any(word in lower for word in ["goal", "target", "aim", "objective"]):
                role = Role.GOAL
            elif any(word in lower for word in ["must", "cannot", "constraint", "limit"]):
                role = Role.CONSTRAINT
            elif any(word in lower for word in ["decided", "chose", "selected"]):
                role = Role.DECISION
            elif any(word in lower for word in ["if", "when", "depends on"]):
                role = Role.CONDITIONAL_DEPENDENCY
            elif any(word in lower for word in ["observed", "noticed", "see"]):
                role = Role.OBSERVATION

            nodes.append(
                Node(
                    text=sentence,
                    role=role,
                    confidence=0.7,
                )
            )

        return nodes

    def __repr__(self) -> str:
        mode = "dummy" if self.use_dummy else "api"
        return f"LLMExtractor(model={self.model}, mode={mode})"
