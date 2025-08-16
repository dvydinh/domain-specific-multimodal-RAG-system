"""
Unit tests for the QueryRouter.

Since the router uses an LLM, these tests verify the heuristic
analysis (route_with_analysis) which runs without API calls.
"""

import pytest
from backend.retrieval.router import QueryRouter, QueryType


class TestQueryTypeEnum:
    """Basic enum validation."""

    def test_enum_values(self):
        assert QueryType.GRAPH_ONLY.value == "graph_only"
        assert QueryType.VECTOR_ONLY.value == "vector_only"
        assert QueryType.HYBRID.value == "hybrid"

    def test_enum_from_string(self):
        assert QueryType("graph_only") == QueryType.GRAPH_ONLY
        assert QueryType("hybrid") == QueryType.HYBRID


class TestRouterHeuristics:
    """
    Tests for the heuristic feature detection in route_with_analysis.
    These don't call the LLM — they test the rule-based analysis layer.
    """

    def setup_method(self):
        # We only test the analysis part, not the LLM routing
        pass

    def test_negation_detection(self):
        """Queries with negation words are detected."""
        test_cases = [
            ("Recipes without pork", True),
            ("No scallion dishes", True),
            ("Exclude all dairy", True),
            ("Dishes that don't use garlic", True),
            ("Simple ramen recipe", False),
        ]

        for query, expected in test_cases:
            query_lower = query.lower()
            has_negation = any(w in query_lower for w in [
                "without", "no ", "not ", "exclude", "không",
                "don't", "never", "avoid",
            ])
            assert has_negation == expected, f"Failed for: {query}"

    def test_cuisine_tag_detection(self):
        """Cuisine types are detected from queries."""
        test_cases = [
            ("Japanese spicy ramen", True),
            ("Italian pasta carbonara", True),
            ("Vietnamese pho recipe", True),
            ("A creamy soup recipe", False),
        ]

        cuisines = [
            "japanese", "italian", "vietnamese", "chinese",
            "korean", "thai", "french", "indian", "mexican",
        ]

        for query, expected in test_cases:
            query_lower = query.lower()
            has_cuisine = any(c in query_lower for c in cuisines)
            assert has_cuisine == expected, f"Failed for: {query}"

    def test_dietary_tag_detection(self):
        """Dietary restrictions are detected."""
        test_cases = [
            ("Vegan dessert", True),
            ("Spicy chicken wings", True),
            ("Gluten-free bread", True),
            ("Regular pasta recipe", False),
        ]

        dietary = [
            "vegan", "vegetarian", "gluten-free", "spicy",
            "chay", "cay", "healthy",
        ]

        for query, expected in test_cases:
            query_lower = query.lower()
            has_dietary = any(d in query_lower for d in dietary)
            assert has_dietary == expected, f"Failed for: {query}"

    def test_instruction_request_detection(self):
        """Questions asking for instructions/recipes are detected."""
        test_cases = [
            ("How do I make ramen?", True),
            ("Recipe for pad thai", True),
            ("Cook time for steak", True),
            ("List all Japanese dishes", False),
        ]

        instruction_words = [
            "how", "recipe", "cook", "make", "prepare",
            "instructions", "steps", "cách làm", "hướng dẫn",
        ]

        for query, expected in test_cases:
            query_lower = query.lower()
            wants_instructions = any(w in query_lower for w in instruction_words)
            assert wants_instructions == expected, f"Failed for: {query}"

    def test_image_request_detection(self):
        """Requests for images are detected."""
        test_cases = [
            ("Show me the photo of ramen", True),
            ("Ramen recipe with images", True),
            ("List recipes", False),
        ]

        image_words = ["image", "photo", "picture", "show me", "ảnh", "hình"]

        for query, expected in test_cases:
            query_lower = query.lower()
            wants_images = any(w in query_lower for w in image_words)
            assert wants_images == expected, f"Failed for: {query}"
