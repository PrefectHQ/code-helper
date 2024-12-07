import tempfile
from httpx import ASGITransport
import pytest_asyncio
import pytest

from prefect.testing.utilities import prefect_test_harness

from code_helper.index import process_file
from code_helper.models import async_drop_db, init_db_connection, init_db

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield


@pytest_asyncio.fixture(autouse=True)
async def test_db():
    """Create a test database and handle setup/teardown"""
    TEST_DB_URL = (
        "postgresql+asyncpg://code_helper:help-me-code@localhost:5432/code_helper_test"
    )
    
    # Initialize test database connection
    test_engine, test_session_local = init_db_connection(TEST_DB_URL)
    
    # Initialize schema
    await init_db(test_engine)
    
    yield test_session_local
    
    # Cleanup
    await async_drop_db(test_engine)
    await test_engine.dispose()


@pytest_asyncio.fixture
async def test_client(test_db):
    from code_helper.app import app
    from httpx import AsyncClient
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


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
        current_weight = sum(c.weight for c in self.cargo)
        if current_weight + cargo.weight > self.max_cargo_weight:
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


@pytest_asyncio.fixture
async def db_session(test_db):
    """Get a database session for testing."""
    async with test_db() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

