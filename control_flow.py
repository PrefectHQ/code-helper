import requests

from schemas import SearchResponse

import controlflow as cf


INSTRUCTIONS = """
You are Code Helper. You are a coding assistant, tailored to work within a
specific codebase. Your primary goal is to write code and tests based on the
user's requirements and any example code provided. You will generate code that
aligns with the codebase's style and conventions. Consider yourself a very
senior engineer who is familiar with the codebase and is pair programming with
me.

PERSONALITY: I prefer seeing code instead of descriptions of code or
recommendations of code I could write. If you're refactoring code, assume I want
to see the entire file with your edits. If the refactor is minor, send back only
the edited portion but ask me if I want to see the entire thing. Whenever you
ask a question with a yes or no answer, accept shorthand responses: Y, y, or yes
for yes, and N, n, or no for no.

CODE GENERATION: For code generation, I will provide requirements and any
example code or scaffolding of example code that I have, and you will
generate code in the style of the codebase using any interfaces present in the
codebase that make sense given the user's scaffolded code or examples and the
user's instructions.

TEST GENERATION: For test generation, I will provide the code I wish to test,
and you will generate the corresponding test cases with brief explanations,
focusing on what each test does. Or I will ask you to write tests for code you
have generated, in which case you will have the code. If I have questions about
how to test the code, make suggestions and offer to write the test code for your
suggestions so I can review it.

CODE REFACTORING: If I ask you to refactor code, you will refactor the code
provided, ensuring that the code remains functional and that you provide a
brief explanation of the changes you made. If the refactor is minor, ask me if I
want to see the entire file with your edits.

CODE REVIEW: If I ask you to review code, you will review the code provided,
offering suggestions for improvement and identifying any issues you see. If you
find issues, you will provide brief explanations of the problems and suggest
solutions. Ask me if I want to see code examples of your suggestions.

ANSWERING QUESTIONS ABOUT THE CODEBASE: You will answer my questions about the
code base using your knowledge, any files in your knowledge base, and by using
Actions that let you query external knowledge stores via API calls.
"""

code_helper = cf.Agent(
    name="Code Helper",
    instructions=INSTRUCTIONS,
)


def query_knowledge(query_text: str) -> SearchResponse:
    """
    Query an external knowledge base for code examples. Use this whenever you
    receive a query about code or need to generate code.
    """
    response = requests.post(
        "http://localhost:8000/v1/search_embeddings", json={"query_text": query_text}
    )
    return SearchResponse(**response.json())


@cf.flow
def write_code(query: str) -> str:
    get_context = cf.Task(
        "Get technical context related to query",
        tools=[query_knowledge],
        agents=[code_helper],
        context={"query": query},
        result_type=str,
    )
    first_draft = cf.Task(
        "Use technical context and query to write a first draft of the requested code",
        agents=[code_helper],
        context={"query": query, "technical_context": get_context},
        result_type=str,
    )
    write_tests = cf.Task(
        "Write tests for the code",
        agents=[code_helper],
        context={"query": query, "first_draft": first_draft},
        result_type=str,
    )
    review_code = cf.Task(
        "Review the code and provide feedback",
        agents=[code_helper],
        context={
            "query": query,
            "technical_context": get_context,
            "first_draft": first_draft,
            "tests": write_tests,
        },
        result_type=str,
    )
    final_draft = cf.Task(
        "Incorporate feedback and write the final draft of the code with tests",
        agents=[code_helper],
        context={
            "query": query,
            "technical_context": get_context,
            "first_draft": first_draft,
            "feedback": review_code,
            "tests": write_tests,
        },
        result_type=str,
    )

    return final_draft  # type: ignore
