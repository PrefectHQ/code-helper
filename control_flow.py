import sys
from typing import Optional

import marvin
import requests
from controlflow import flow
from prefect import pause_flow_run
from prefect.context import EngineContext
from prefect.input import RunInput
from prefect.utilities.urls import url_for

from schemas import SearchResponse
from dotenv import load_dotenv

import controlflow as cf

load_dotenv()

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


class Interaction(RunInput):
    query: str


@cf.flow(agents=[code_helper])
def write_code(query: Optional[str] = None) -> str:
    get_context = cf.Task(
        "Get technical context related to query",
        tools=[query_knowledge],
        context={"query": query},
    )
    first_draft = cf.Task(
        "Use technical context and query to write a first draft of the requested code",
        context={"query": query, "technical_context": get_context},
    )
    write_tests = cf.Task(
        "Write tests for the code",
        context={"query": query, "first_draft": first_draft},
    )
    review_code = cf.Task(
        "Review the code and provide feedback",
        context={
            "query": query,
            "technical_context": get_context,
            "first_draft": first_draft,
            "tests": write_tests,
        },
    )
    final_draft = cf.Task(
        "Incorporate feedback and write the final draft of the code with tests. "
        "Wrap any code in a markdown code block.",
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


@cf.flow(agents=[code_helper])
def review_code(query: Optional[str] = None) -> str:
    get_context = cf.Task(
        "Get technical context related to query",
        tools=[query_knowledge],
        context={"query": query},
        result_type=str,
    )
    review_code = cf.Task(
        "Review the code and provide feedback",
        context={
            "query": query,
            "technical_context": get_context,
        },
        result_type=str,
    )
    return review_code  # type: ignore


@cf.flow(agents=[code_helper])
def write_tests(query: Optional[str] = None) -> str:
    get_context = cf.Task(
        "Get technical context related to query",
        tools=[query_knowledge],
        context={"query": query},
    )
    write_tests = cf.Task(
        "Use technical context to write tests for the code",
        context={"query": query, "technical_context": get_context},
    )
    return write_tests  # type: ignore


@cf.flow(agents=[code_helper])
def refactor_code(query: Optional[str] = None) -> str:
    get_context = cf.Task(
        "Get technical context related to query",
        tools=[query_knowledge],
        context={"query": query},
    )
    refactor_code = cf.Task(
        "Refactor the code",
        context={"query": query, "technical_context": get_context},
    )
    return refactor_code  # type: ignore


@cf.flow(agents=[code_helper])
def answer_question(query: Optional[str] = None) -> str:
    get_context = cf.Task(
        "Get technical context related to query",
        tools=[query_knowledge],
        context={"query": query},
    )
    answer_question = cf.Task(
        "Answer the question",
        context={"query": query, "technical_context": get_context},
    )
    return answer_question  # type: ignore


@cf.flow(agents=[code_helper])
def answer_unclassifiable_query(query: Optional[str] = None) -> str:
    get_context = cf.Task(
        "Get technical context related to query",
        tools=[query_knowledge],
        context={"query": query},
    )
    try_to_answer_unclassified_query = cf.Task(
        "Try to answer the unclassifiable query. Mention that you aren't sure "
        "exactly how to answer the question, but you'll give it your best shot.",
        context={"query": query, "technical_context": get_context},
    )
    return try_to_answer_unclassified_query  # type: ignore


class Unset:
    pass


def classify(query: str) -> str:
    return marvin.classify(
        query,
        [
            "write_code",
            "write_tests",
            "refactor_code",
            "review_code",
            "answer_question",
            "unclassifiable",
        ],
    )


def answer_query(query: str) -> str:
    workflow = classify(query)
    if workflow == "write_code":
        return write_code(query=query)
    elif workflow == "write_tests":
        return write_tests(query=query)
    elif workflow == "refactor_code":
        return refactor_code(query=query)
    elif workflow == "review_code":
        return review_code(query=query)
    elif workflow == "answer_question":
        return answer_question(query=query)
    elif workflow == "unclassifiable":
        return answer_unclassifiable_query(query=query)
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


@flow
def write_code_loop(query: Optional[str] = None) -> str:
    if query is None:  # Prefect parameter validation can't handle Unset
        query = Unset

    engine_ctx = EngineContext.get()
    flow_run_url = url_for(engine_ctx.flow_run)

    try:
        while True:
            if query is Unset:
                response = "How can I help?"
            elif query is None or query == "q" or query == "quit":
                return "Goodbye!"
            else:
                print("Getting response")
                response = answer_query(query=query)

            print("Pausing for query. View at: ", flow_run_url)
            interaction = pause_flow_run(
                Interaction.with_initial_data(description=response)
            )
            query = interaction.query
    except KeyboardInterrupt:
        return "Goodbye!"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python control_flow.py [run|serve]")
        sys.exit(1)
    if sys.argv[1] == "run":
        write_code_loop()
    elif sys.argv[1] == "serve":
        write_code_loop.serve()
