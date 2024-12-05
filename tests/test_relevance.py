import collections
from pprint import pprint
import pytest

from code_helper.index import generate_embeddings
from code_helper.models import hybrid_search


@pytest.mark.asyncio
async def test_relevance_precision(db_session, files):
    spaceship_file, _, astronaut_file, test_file, utility_file = files
    
    # Test cases for different types of queries
    test_cases = [
        {
            "query": "How do I check if a mission is ready to launch?",
            "expected_matches": [{
                "function": "check_mission_readiness",
                "type": "function",
                "file": utility_file
            }]
        },
        {
            "query": "Show me code related to monitoring active space missions",
            "expected_matches": [{
                "function": "monitor_mission",
                "type": "method",
                "class": "MissionControl",
                "file": utility_file
            }]
        },
        {
            "query": "How can I validate crew member expertise?",
            "expected_matches": [
                {
                    "function": "validate_crew_expertise",
                    "type": "function",
                    "file": utility_file
                },
                {
                    "function": "check_mission_readiness",
                    "type": "function",
                    "file": utility_file
                }
            ]
        },
        {
            "query": "Calculate total weight of cargo",
            "expected_matches": [{
                "function": "calculate_total_weight",
                "type": "function",
                # "context": "utility",
                "file": utility_file
            }]
        },
        {
            "query": "Show me all crew-related functionality",
            "expected_matches": [
                {
                    "function": "load_crew",
                    "type": "method",
                    "class": "Spaceship",
                    "file": spaceship_file
                },
                {
                    "function": "validate_crew_expertise",
                    "type": "function",
                    "file": utility_file
                }
            ]
        },
        {
            "query": "Find code about launching spaceships",
            "expected_matches": [
                {
                    "function": "launch",
                    "type": "method",
                    "class": "Spaceship",
                    "file": spaceship_file
                },
                {
                    "function": "check_mission_readiness",
                    "type": "function",
                    # "context": "utility",
                    "file": utility_file
                }

            ]
        }
    ]

    for test_case in test_cases:
        if test_case["query"] == "How can I validate crew member expertise?":
            print("hey")
 
        query_vector = await generate_embeddings(test_case["query"])
        results = await hybrid_search(db_session, test_case["query"], query_vector, limit=10)
               
        assert len(results) > 0, f"No results found for query: {test_case['query']}"
        
        # Convert results to a more easily searchable format
        found_matches = [{
            "function": r["metadata"]["name"],
            "type": r["metadata"]["type"],
            "class": r["metadata"].get("parent"),
            "context": r["metadata"].get("context")
        } for r in results]
        
        # Check each expected match
        for expected_match in test_case["expected_matches"]:
            matching_result = next(
                (r for r in found_matches 
                 if r["function"] == expected_match["function"]
                 and r["type"] == expected_match["type"]
                 and (r["class"] == expected_match.get("class") if "class" in expected_match else True)
                 and (r["context"] == expected_match.get("context") if "context" in expected_match else True)),
                None
            )
            
            assert matching_result is not None, (
                f"Expected to find {expected_match} in results for query: {test_case['query']}\n"
                f"Found matches: {found_matches}"
            )

@pytest.mark.asyncio
async def test_relevance_ranking(db_session, files):
    # Test ranking with different query variations
    ranking_tests = [
        {
            "query": "How do I add crew members to a spaceship?",
            "expected_order": [
                "class Spaceship",
                "def load_crew",
                "def validate_crew_expertise",
            ]
        },
        {
            "query": "What's the process for launching a spaceship?",
            "expected_order": [
                ".launch",
                ".is_launched",
                "def launch",
            ]
        },
        {
            "query": "Show me cargo management code",
            "expected_order": [
                "def load_cargo",  # Should find cargo loading first
                "class Spaceship",  # Spaceships work with cargo
                "cargo.weight"  # Usage example
            ]
        }
    ]

    def find_content_index(content_list, search_string):
        """Find the first result that contains the search string."""
        for idx, content in enumerate(content_list):
            if search_string.lower() in content.lower():
                return idx
        return -1

    for test in ranking_tests:
        query_vector = await generate_embeddings(test["query"])
        results = await hybrid_search(db_session, test["query"], query_vector, limit=10)
        
        found_contents = [r["content"] for r in results]
        
        # Check if strings appear in content at or before their maximum allowed position
        for i, expected_string in enumerate(test["expected_order"]):
            found_idx = find_content_index(found_contents, expected_string)
            assert found_idx != -1, (
                f"Expected string '{expected_string}' not found in any result content for query: {test['query']}\n"
                f"Available contents:\n" + "\n".join(f"{i}: {c}" for i, c in enumerate(found_contents))
            )
            max_allowed_position = len(test["expected_order"]) - 1
            assert found_idx <= max_allowed_position, (
                f"Expected '{expected_string}' to appear at or before position {max_allowed_position}\n"
                f"Found '{expected_string}' at position {found_idx}\n"
                f"Results order:\n" + "\n".join(f"{i}: {c}" for i, c in enumerate(found_contents))
            )


@pytest.mark.asyncio
async def test_relevance_context_awareness(db_session, files):
    # Test that results include relevant context
    context_tests = [
        {
            "query": "Show me the implementation of crew loading with capacity checks",
            "expected_content": ["crew_capacity", "load_crew", "cannot load"],
            "expected_metadata": {"type": "method", "name": "load_crew"}
        },
        {
            "query": "Refactor weight checks for cargo loading",
            "expected_content": ["max_cargo_weight", "load_cargo", "cannot load"],
            "expected_metadata": {"type": "method", "name": "load_cargo"}
        }
    ]

    for test in context_tests:
        query_vector = await generate_embeddings(test["query"])
        results = await hybrid_search(db_session, test["query"], query_vector, limit=5)
        
        top_results = results[:5]
        
        # Check that the top result contains expected content
        for expected_content in test["expected_content"]:
            found = False
            for result in top_results:
                found = expected_content.lower() in result["content"].lower()
                if found:
                    break
            assert found, \
                f"Expected content '{expected_content}' not found in top result"
        
        # Verify metadata matches expectations
        for key, value in test["expected_metadata"].items():
            found = False
            for result in top_results:
                found = result["metadata"][key] == value
                if found:
                    break
            assert found, \
                f"Expected metadata {key}={value} not found in top result"


@pytest.mark.asyncio
async def test_relevance_no_duplicates(db_session, files):
    queries = [
        "How do I check if a mission is ready to launch?",
        "Show me code related to monitoring active space missions",
        "How can I validate crew member expertise?",
        "Calculate total weight of cargo",
        "Show me all crew-related functionality",
        "Find code about launching spaceships"
    ]

    for query in queries:
        query_vector = await generate_embeddings(query)
        results = await hybrid_search(db_session, query, query_vector, limit=10)
        # Test that there are no duplicate combinations of name and type
        combinations = [(r["metadata"]["name"], r["metadata"]["type"], r["metadata"].get("parent")) for r in results]    
        duplicates = [item for item, count in collections.Counter(combinations).items() if count > 1]
        assert len(duplicates) == 0, \
            f"Duplicate combinations of name and type found for query: {query}. Duplicates: {duplicates}, results: {pprint(results)}"

