from logging import getLogger
import sys
from typing import Optional, Any
from pydantic import Field

import requests
from marvin.beta import Assistant
from marvin.beta.assistants.threads import Thread
from marvin.beta.assistants.assistants import NOT_PROVIDED, default_run_handler_class
from marvin.tools.assistants import CodeInterpreter
from marvin.types import Run
from marvin.utilities.asyncio import expose_sync_method
from openai.lib.streaming import AsyncAssistantEventHandler

from schemas import SearchResponse

logger = getLogger(__name__)


def visit_url(url: str):
    return requests.get(url).content.decode()


def query_knowledge(query_text: str) -> SearchResponse:
    """
    Query an external knowledge base for code examples. Use this whenever you
    receive a query about code or need to generate code.
    """
    response = requests.post(
        "http://localhost:8000/v1/search_embeddings", json={"query_text": query_text}
    )
    return SearchResponse(**response.json())


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

# The assistant won't use the query_knowledge tool unless it's explicitly
# reminded to do so. To avoid having to remember to tell it every time the code
# search API would be relevant, we add a reminder to every message.
SPECIAL_INSTRUCTION = """
[For Code Helper]: Remember to query your knowledge whenever answering questions
about code, reviewing code, or generating code, even if you think you know the
answer.
"""


class CodeAssistant(Assistant):
    @expose_sync_method("say")
    async def say_async(
        self,
        message: str,
        code_interpreter_files: Optional[list[str]] = None,
        file_search_files: Optional[list[str]] = None,
        thread: Optional[Thread] = None,
        event_handler_class: type[AsyncAssistantEventHandler] = NOT_PROVIDED,
        **run_kwargs,
    ) -> "Run":
        thread = thread or self.default_thread

        if event_handler_class is NOT_PROVIDED:
            event_handler_class = default_run_handler_class()

        enhanced_message = f"{message}\n{SPECIAL_INSTRUCTION}"

        # post the message
        user_message = await thread.add_async(
            enhanced_message,
            code_interpreter_files=code_interpreter_files,
            file_search_files=file_search_files,
        )

        from marvin.beta.assistants.runs import Run

        run = Run(
            # provide the user message as part of the run to print
            messages=[user_message],
            assistant=self,
            thread=thread,
            event_handler_class=event_handler_class,
            **run_kwargs,
        )
        result = await run.run_async()
        return result




code_helper = CodeAssistant(
    name="Code Helper",
    tools=[CodeInterpreter, visit_url, query_knowledge],
    description="A coding assistant for writing code and tests",
    instructions=INSTRUCTIONS,
)

if __name__ == "__main__":
    try:
        query = sys.argv[1]
    except IndexError:
        query = "hi"

    code_helper.chat(query)
