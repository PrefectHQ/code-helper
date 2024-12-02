# Use MAPS to test relevance. Index multiple sample files 
# and then test the relevance of queries against the index.

import collections
from pprint import pprint
import tempfile
import pytest

from code_helper.index import generate_embeddings, process_file
from code_helper.models import hybrid_search

TEST_SPACESHIP_FILE_CONTENT = """
class Spaceship:
    def __init__(self, name: str, fuel_level: int = 100, is_launched: bool = False, cargo: list["Cargo"] = [], max_cargo_weight: int = 1000, crew: list["Astronaut"] = [], crew_capacity: int = 10):
        self.name = name
        self.fuel_level = fuel_level
        self.is_launched = is_launched
        self.cargo = cargo
        self.max_cargo_weight = max_cargo_weight
        self.crew = crew
        self.crew_capacity = crew_capacity

    def launch(self):
        if self.fuel_level > 0:
            self.is_launched = True
            print(f"{self.name} launched!")
        else:
            print(f"{self.name} does not have enough fuel to launch.")
            
    def land(self):
        if self.is_launched:
            self.is_launched = False
            print(f"{self.name} landed!")
        else:
            print(f"{self.name} is not launched yet.")  
            
    def load_cargo(self, cargo: "Cargo"):
        if self.cargo_weight + cargo.weight > self.max_cargo_weight:
            print(f"{self.name} cannot load {cargo.name} because it would exceed the max cargo weight.")
        else:
            self.cargo.append(cargo)
            print(f"{self.name} loaded {cargo.name}.")
            
    def load_crew(self, crew: "Astronaut"):
        if len(self.crew) + 1 > self.crew_capacity:
            print(f"{self.name} cannot load {crew.name} because it would exceed the crew capacity.")
        else:
            self.crew.append(crew)
            print(f"{self.name} loaded {crew.name}.")
"""

TEST_CARGO_FILE_CONTENT = """
class Cargo:
    def __init__(self, name: str, weight: int = 0):
        self.name = name
        self.weight = weight
"""


TEST_ASTRONAUT_FILE_CONTENT = """
class Expertise(Enum):
    MECHANICAL = "Mechanical"
    ELECTRICAL = "Electrical"
    SOFTWARE = "Software"


class Astronaut:
    def __init__(self, name: str, skill_level: int = 0, expertise: Expertise = Expertise.MECHANICAL):
        self.name = name
        self.skill_level = skill_level
        self.expertise = expertise
"""


TEST_TEST_FILE_CONTENT = """
def test_spaceship():
    spaceship = Spaceship("Apollo")
    spaceship.launch()
    assert spaceship.is_launched == True
    

def test_cargo():
    cargo = Cargo("Toolbox", 10)
    assert cargo.weight == 10
    

def test_astronaut():
    astronaut = Astronaut("John", 5, Expertise.MECHANICAL)
    assert astronaut.name == "John"
    assert astronaut.skill_level == 5
    assert astronaut.expertise == Expertise.MECHANICAL
    

def test_spaceship_with_cargo():
    spaceship = Spaceship("Apollo")
    cargo = Cargo("Toolbox", 10)
    spaceship.load_cargo(cargo)
    assert len(spaceship.cargo) == 1
    assert spaceship.cargo[0].name == "Toolbox"
"""

# Add more test content for better coverage
TEST_UTILITY_FILE_CONTENT = """
def calculate_total_weight(cargo_list: List["Cargo"]) -> int:
    return sum(cargo.weight for cargo in cargo_list)

def validate_crew_expertise(crew_member: "Astronaut", required_expertise: "Expertise") -> bool:
    return crew_member.expertise == required_expertise

def check_mission_readiness(spaceship: "Spaceship") -> bool:
    if spaceship.fuel_level < 20:
        return False
    if len(spaceship.crew) < 2:
        return False
    return True

class MissionControl:
    def __init__(self):
        self.active_missions = []
        
    def monitor_mission(self, spaceship: "Spaceship"):
        if spaceship.is_launched:
            self.active_missions.append(spaceship)
            
    def get_mission_status(self, spaceship: "Spaceship") -> str:
        if spaceship in self.active_missions:
            return "Active"
        return "Inactive"
"""


@pytest.fixture
async def files(db_session):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(TEST_SPACESHIP_FILE_CONTENT.encode("utf-8"))
        spaceship_file = f.name
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(TEST_CARGO_FILE_CONTENT.encode("utf-8"))
        cargo_file = f.name
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(TEST_ASTRONAUT_FILE_CONTENT.encode("utf-8"))
        astronaut_file = f.name
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(TEST_TEST_FILE_CONTENT.encode("utf-8"))
        test_file = f.name
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(TEST_UTILITY_FILE_CONTENT.encode("utf-8"))
        utility_file = f.name

    # Process all files
    await process_file(spaceship_file, db_session)
    await process_file(cargo_file, db_session)
    await process_file(astronaut_file, db_session)
    await process_file(test_file, db_session)
    await process_file(utility_file, db_session)

    return spaceship_file, cargo_file, astronaut_file, test_file, utility_file


@pytest.mark.asyncio
async def test_relevance_precision(db_session, files):
    spaceship_file, cargo_file, astronaut_file, test_file, utility_file = files
    
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
            "expected_order": ["load_crew", "Astronaut", "crew_capacity"]
        },
        {
            "query": "What's the process for launching a spaceship?",
            "expected_order": ["launch", "is_launched", "fuel_level"]
        },
        {
            "query": "Show me cargo management code",
            "expected_order": ["load_cargo", "Cargo", "max_cargo_weight"]
        }
    ]

    for test in ranking_tests:
        query_vector = await generate_embeddings(test["query"])
        results = await hybrid_search(db_session, test["query"], query_vector, limit=10)
        
        found_items = [r["metadata"]["name"] for r in results]
        
        # Check if items appear in the expected order
        last_found_idx = -1
        for expected_item in test["expected_order"]:
            try:
                found_idx = found_items.index(expected_item)
                assert found_idx > last_found_idx, \
                    f"Expected {expected_item} to appear after {test['expected_order'][last_found_idx]}"
                last_found_idx = found_idx
            except ValueError:
                assert False, f"Expected item {expected_item} not found in results"

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
            "query": "How does cargo weight validation work?",
            "expected_content": ["max_cargo_weight", "load_cargo", "cannot load"],
            "expected_metadata": {"type": "method", "name": "load_cargo"}
        }
    ]

    for test in context_tests:
        query_vector = await generate_embeddings(test["query"])
        results = await hybrid_search(db_session, test["query"], query_vector, limit=5)
        
        top_result = results[0]
        
        # Check that the top result contains expected content
        for expected_content in test["expected_content"]:
            assert expected_content.lower() in top_result["fragment_content"].lower(), \
                f"Expected content '{expected_content}' not found in top result"
        
        # Verify metadata matches expectations
        for key, value in test["expected_metadata"].items():
            assert top_result["metadata"][key] == value, \
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

