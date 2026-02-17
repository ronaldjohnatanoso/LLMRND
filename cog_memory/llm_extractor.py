"""LLM-based commitment extraction from text.

Extracts commitments with meta-roles from paragraphs using LLM.

Supports multiple providers:
- OpenAI: GPT-4o, GPT-4o-mini
- Groq: Llama 3.1 8B (FREE), Llama 3.1 70B (FREE)
- Hugging Face: Serverless inference (free tier)
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv

from cog_memory.node import Node, Role

if TYPE_CHECKING:
    from collections.abc import Sequence


load_dotenv()

Provider = Literal["openai", "groq", "huggingface"]


class LLMExtractor:
    """Extract commitments from text using LLM.

    Parses paragraphs into structured commitments with meta-role assignments.

    Supports multiple inference providers:
    - **Groq** (recommended for testing): FREE, fastest inference
      Get API key: https://console.groq.com/
      Models: llama-3.1-8b-instant, llama-3.1-70b-versatile
    - **OpenAI**: GPT-4o, GPT-4o-mini (requires paid API)
    - **Hugging Face**: Serverless API with free tier
    """

    # Default models per provider
    DEFAULT_MODELS = {
        "groq": "llama-3.1-8b-instant",
        "openai": "gpt-4o",
        "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
    }

    # Base URLs for each provider
    BASE_URLS = {
        "groq": "https://api.groq.com/openai/v1",
        "openai": None,  # Use OpenAI default
        "huggingface": "https://api-inference.huggingface.co",
    }

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.3,
        use_dummy: bool = False,
        provider: Provider = "groq",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the LLM extractor.

        Args:
            model: Model name (uses provider default if None)
            temperature: Sampling temperature for generation
            use_dummy: Use dummy extractor for testing (no API calls)
            provider: Provider to use ('groq', 'openai', 'huggingface')
            api_key: API key (uses env var if None)
            base_url: Custom base URL (uses provider default if None)
        """
        self.provider = provider
        self.temperature = temperature
        self.use_dummy = use_dummy

        # Set model default based on provider
        if model is None:
            model = self.DEFAULT_MODELS.get(provider, "gpt-4o")
        self.model = model

        # Set API key
        if api_key is None:
            if provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "huggingface":
                api_key = os.getenv("HUGGINGFACE_API_KEY")

        if not use_dummy:
            from openai import OpenAI

            # Set base URL
            if base_url is None:
                base_url = self.BASE_URLS.get(provider)

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

    def _build_extraction_prompt(self, text: str) -> str:
        """Build the prompt for commitment extraction.

        Args:
            text: Input text to extract commitments from

        Returns:
            Formatted prompt for the LLM
        """
        return f"""Extract meaningful commitments from the following text and assign each a meta-role.

Meta-roles:
- fact: Verifiable information
- observation: Noted information without verification
- goal: Target state or objective to achieve
- constraint: Limitation or restriction on actions
- decision: Chosen course of action
- conditional_dependency: Relationship between nodes (if X then Y)

CRITICAL REQUIREMENTS:
- Extract ONLY complete sentences that express a full thought
- Do NOT break sentences into fragments
- Do NOT extract partial phrases or segments
- Each commitment MUST be a standalone, meaningful statement
- Prioritize quality over quantity - better to miss a commitment than extract a bad one

Examples of GOOD extractions:
Input: "The system requires 50 TB of storage and must maintain 99.99% uptime."
✓ Extract: "The system requires 50 TB of storage" (fact)
✓ Extract: "The system must maintain 99.99% uptime" (goal)

Input: "banana is the best fucking food in the whole universe"
✓ Extract: "banana is the best fucking food in the whole universe" (observation)
✗ DO NOT extract: "banana is the best"
✗ DO NOT extract: "the whole universe"
✗ DO NOT extract: "the best fucking food"

Input: "Due to budget constraints, we cannot use enterprise tools and must use open source alternatives."
✓ Extract: "Due to budget constraints, we cannot use enterprise tools" (constraint)
✓ Extract: "we must use open source alternatives" (decision)
✗ DO NOT extract: "budget constraints"
✗ DO NOT extract: "enterprise tools"

For each commitment, provide:
1. text: A complete, standalone sentence (minimum 15 characters, complete thought)
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
            # Hugging Face uses a different API format
            if self.provider == "huggingface":
                return self._extract_huggingface(prompt)

            # OpenAI-compatible API (Groq, OpenAI)
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
                response_format={"type": "json_object"} if self.provider == "openai" else None,
            )

            result = response.choices[0].message.content
            if not result:
                return []

            # Parse JSON response (may have markdown code blocks)
            result = self._clean_json_response(result)
            data = json.loads(result)

            # Handle both dict with "commitments" key and direct list
            if isinstance(data, dict):
                commitments = data.get("commitments", data)
            else:
                commitments = data

            nodes = []
            for item in commitments:
                node_text = item.get("text", "").strip()

                # Validate minimum length (15 characters)
                if len(node_text) < 15:
                    print(f"⚠️ Skipping short commitment: '{node_text[:50]}...' ({len(node_text)} chars)")
                    continue

                # Validate meaningful content (not just repeated words)
                words = node_text.split()
                if len(words) < 3:
                    print(f"⚠️ Skipping commitment with too few words: '{node_text[:50]}...'")
                    continue

                nodes.append(
                    Node(
                        text=node_text,
                        role=Role(item["role"]),
                        confidence=item.get("confidence", 0.7),
                    )
                )

            if not nodes:
                print("⚠️ No valid commitments found after filtering")

            return nodes

        except Exception as e:
            print(f"Error extracting commitments: {e}")
            return self._dummy_extract(text)

    def _extract_huggingface(self, prompt: str) -> list[Node]:
        """Extract using Hugging Face inference API.

        Args:
            prompt: The extraction prompt

        Returns:
            List of Node objects
        """
        import requests

        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise commitment extractor. Always return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 2048,
        }

        response = requests.post(
            f"{self.client.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        if not content:
            return []

        # Clean and parse JSON
        content = self._clean_json_response(content)
        data = json.loads(content)

        # Handle both dict with "commitments" key and direct list
        if isinstance(data, dict):
            commitments = data.get("commitments", data)
        else:
            commitments = data

        nodes = []
        for item in commitments:
            node_text = item.get("text", "").strip()

            # Validate minimum length (15 characters)
            if len(node_text) < 15:
                print(f"⚠️ Skipping short commitment: '{node_text[:50]}...' ({len(node_text)} chars)")
                continue

            # Validate meaningful content (not just repeated words)
            words = node_text.split()
            if len(words) < 3:
                print(f"⚠️ Skipping commitment with too few words: '{node_text[:50]}...'")
                continue

            nodes.append(
                Node(
                    text=node_text,
                    role=Role(item["role"]),
                    confidence=item.get("confidence", 0.7),
                )
            )

        if not nodes:
            print("⚠️ No valid commitments found after filtering")

        return nodes

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response that may contain markdown code blocks.

        Args:
            response: Raw response string

        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            # Skip first line (```json or ```) and last line (```)
            if len(lines) > 2:
                response = "\n".join(lines[1:-1])

        return response.strip()

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
        mode = "dummy" if self.use_dummy else self.provider
        return f"LLMExtractor(provider={mode}, model={self.model})"
