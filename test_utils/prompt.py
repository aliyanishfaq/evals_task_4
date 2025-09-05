LLM_AS_A_JUDGE_PROMPT = '''
You are an expert coding evaluator for agent that stores memories built using langgraph. You will be provided with a coding agent's implementation and an expert-written implementation that represents the gold standard, alongside the base code that was provided to the coding agent to start with.

The coding agent was provided with a task to modify the existing memory agent code to store memories user id and category wise and have interrupt before saving a memory. Your job is to evaluate the coding agent's implementation for correctness, code quality, and adherence to the expert's architectural and code cleanliness philosophy. 
You will be provided with basic requirements and good practices. You need to check for presence of each of the basic requirements and good practices.
Additionally, you will be asked to evaluate the code for correctness and quality. For correctness error, you will be expected to add an entry to the code_correctness_evidence array.

**CRITICAL RULE - NO DUPLICATE ISSUES**: Never duplicate the same issue across multiple evidence arrays. Each specific issue should only appear ONCE in the entire evaluation - either in code_correctness_evidence OR code_quality_evidence, but never both. If an issue affects both correctness and quality, choose the most appropriate category and only include it there.

**CRITICAL RULE - NO CONTRADICTIONS**: If you mark a basic requirement as `true`, do NOT add contradictory evidence claiming it's missing or broken. The basic requirements and evidence arrays must be logically consistent.

Return your evaluation as a single JSON object.

## Evaluation Criteria

Basic Requirements: The code must meet these core specifications for a functional implementation of edited Memory Agent. Mark `true` if present and working correctly, `false` if absent or broken.

-  presence_of_interrupt: The code should the interrupt functionality in the tool call node before saving a memory. Presence of interrupt is sufficient to award this point.
-  user_id_and_category_wise_storage: The code should have a functionally correct implementation of storing memories category wise. Presence of this is sufficient to award this point.
-  correct_categories: The code should only store memories in the correct three categories. Personal, Professional, Other. Point should only be award if the code is correct in terms of logic and the categories are stored correctly. If there are any other categories or no categories are stored, the point should not be awarded.
-  category_retrieval: When code load memories, it should retrieve memories user id and category wise. Presence of this is sufficient to award this point. If the code does not look for category when retrieving memories, the point should not be awarded.

Good Practices: Additional qualities of a well-designed Memory Agent for this task. Mark `true` if implemented correctly, `false` if absent.


- functional_interrupt: The code should have a functionally correct implementation of the interrupt functionality in the tool call node before saving a memory. The implementation should be correct to be awarded full point.
The simplest implementation for this is the one used by the expert code. If an implementation is like that, then award True. Award true for other correct implementations that save memories on user's input being accept and reject on
everything else.
- llm_based_categorization: The code should use the LLM to categorize the memories. Other implementations such as hardcoding, string/keyword matching are britter and this should be marked false.
- no_test_files: There should be no test files in the code. The code should be as clean and concise as possible. Return false if there are files like `test_agent.py`, `demo.py`

## Evidence Categorization Guidelines

**Code Quality Evidence** (architectural/design issues):
- Separate prompt in the prompt file for llm classification.
- No unnecessary complexity
- Correct primitives for Langgraph types such as state, messages (which should be defined as Annotated[list, add_messages] in state), context, etc.

**Code Correctness Evidence** (functional/runtime bugs):
- No Runtime exceptions or type errors
- Type errors, import failures

**Remember**: NEVER put the same issue in both categories. Choose the most appropriate one.

## JSON Response Structure

{{
  "basic_requirements": {{
    "presence_of_interrupt": boolean,
    "user_id_and_category_wise_storage": boolean,
    "correct_categories": boolean,
    "category_retrieval": boolean,
  }},
  "good_practices": {{
    "functional_interrupt": boolean,
    "llm_based_categorization": boolean,
    "no_test_files": boolean,
  }},
  "code_quality_check": boolean,
  "code_quality_evidence": [
    {{ "issue": string, "severity": "minor|major|critical" }}
  ],
  "code_correctness_check": boolean,
  "code_correctness_evidence": [
    {{ "issue": string, "severity": "minor|major|critical" }}
  ]
}}

Task given to coding agent:
{user_task}

Expert implementation (gold standard):
{expert_code}


Base code provided to the coding agent:
{base_code}


Coding agent implementation to evaluate:
{user_code}

The human annotator added the following notes when running the code:
{human_notes}
'''

USER_TASK = '''
You have been provided code for a memory agent. I want to extend its functionality. Right now, it stores memories user id wise. 
I want it to store user id and category wise. While storing a memory it should determine if a memory is personal, professional, or other. 
It should then save based on that. While retrieving memories it should determine what category the message(s) are from. Based on that, it should only retrieve relevant memories.
Additionaly, there should be an interrupt before saving a memory. If the user inputs accept, the memory should get saved. If the user inputs anything else, the memory should be rejected.
Fix any other bugs/issues as well.
'''

EXPERT_CODE = '''
File Name: context.py

-----------------------------

File Content: 

"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))




File Name: graph.py

-----------------------------

File Content: 

"""Graphs that extract memories on a schedule."""

import asyncio
import logging
from datetime import datetime
from typing import cast

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)

# Initialize the language model to be used for memory extraction
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    user_id = runtime.context.user_id
    model = runtime.context.model
    system_prompt = runtime.context.system_prompt

    category = await utils.get_memory_category(state.messages, llm)

    # Retrieve the most recent memories for context
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id, category),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format memories for inclusion in the prompt
    formatted = "\n".join(
        f"[{mem.key}]: {{mem.value}} (similarity: {{mem.score}})" for mem in memories
    )
    if formatted:
        formatted = f"""
<memories>
{{formatted}}
</memories>"""

    # Prepare the system prompt with user memories and current time
    # This helps the model understand the context and temporal relevance
    sys = system_prompt.format(user_info=formatted, time=datetime.now().isoformat())

    # Invoke the language model with the prepared prompt and tools
    # "bind_tools" gives the LLM the JSON schema for all tools in the list so it knows how
    # to use them.
    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages]
    )
    return {"messages": [msg]}


async def store_memory(state: State, runtime: Runtime[Context]):
    # Extract tool calls from the last message
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    # Concurrently execute all upsert_memory calls
    saved_memories = await asyncio.gather(
        *(
            tools.upsert_memory(
                **tc["args"],
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store),
            )
            for tc in tool_calls
        )
    )

    # Format the results of memory storage operations
    # This provides confirmation to the model that the actions it took were completed
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        # If there are tool calls, we need to store memories
        return "store_memory"
    # Otherwise, finish; user can send the next message
    return END


# Create the graph + all nodes
builder = StateGraph(State, context_schema=Context)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response
builder.add_edge("store_memory", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"

app = graph


__all__ = ["graph"]




File Name: prompts.py

-----------------------------

File Content: 

"""Define default prompts."""

SYSTEM_PROMPT = """You are a helpful and friendly chatbot. Get to know the user! \
Ask questions! Be spontaneous! 
{{user_info}}

System Time: {{time}}"""


CATEGORY_PROMPT = """
Based on these recent messages: {{messages}}
    
Which memory categories are relevant? Choose from: personal, professional, other
    
    Respond with only the category name.
    Examples:
    - "professional" (for work discussions)
    - "personal" (for personal topics)  
    - "other" (for other topics)

Your response should always be one word from personal, professional, or other. 
"""



File Name: state.py

-----------------------------

File Content: 

"""Define the shared values."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass(kw_only=True)
class State:
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


__all__ = [
    "State",
]




File Name: tools.py

-----------------------------

File Content: 

"""Define he agent's tools."""

import uuid
from typing import Annotated, Literal, Optional

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore
from langgraph.types import interrupt


async def upsert_memory(
    content: str,
    context: str,
    category: Literal["personal", "professional", "other"],
    *,
    memory_id: Optional[uuid.UUID] = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        category: The category the memory belongs to. Options are:
            - personal -> personal preferences, hobbies, relationships, interests, etc.
            - professional -> work related information, skills, achievements, etc.
            - other -> memories that don't fit into the personal or professional categories.
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    response = interrupt(f"Saving the following memory: {{content}} in the category: {{category}}. Please reply with 'accept' or 'reject'")
    if response == "accept":
        pass
    else:
        return f"Rejected memory: {{content}} in the category: {{category}}"

    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id, category),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"




File Name: utils.py

-----------------------------

File Content: 

"""Utility functions used in our graph."""

from langchain_core.messages import HumanMessage
from typing import Literal
from .prompts import CATEGORY_PROMPT


def split_model_and_provider(fully_specified_name: str) -> dict:
    """Initialize the configured chat model."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return {"model": model, "provider": provider}


async def get_memory_category(messages, llm) -> Literal["personal", "professional", "other"]:
    """Get the category of the memory based on the messages."""

    try:
        recent_messages = [m.content for m in messages[-3:]]

        category_prompt = CATEGORY_PROMPT.format(messages=recent_messages)

        response = await llm.ainvoke([HumanMessage(content=category_prompt)])

        category = response.content.strip()

        if category not in ["personal", "professional", "other"]:
            return "personal"
    except Exception as e:
        return "personal"

    return category
'''

BASE_CODE = '''
File Name: context.py

-----------------------------

File Content: 

"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))




File Name: graph.py

-----------------------------

File Content: 

"""Graphs that extract memories on a schedule."""

import asyncio
import logging
from datetime import datetime
from typing import cast

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)

# Initialize the language model to be used for memory extraction
llm = init_chat_model()


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    user_id = runtime.context.user_id
    model = runtime.context.model
    system_prompt = runtime.context.system_prompt

    # Retrieve the most recent memories for context
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format memories for inclusion in the prompt
    formatted = "\n".join(
        f"[{mem.key}]: {{mem.value}} (similarity: {{mem.score}})" for mem in memories
    )
    if formatted:
        formatted = f"""
<memories>
{{formatted}}
</memories>"""

    # Prepare the system prompt with user memories and current time
    # This helps the model understand the context and temporal relevance
    sys = system_prompt.format(user_info=formatted, time=datetime.now().isoformat())

    # Invoke the language model with the prepared prompt and tools
    # "bind_tools" gives the LLM the JSON schema for all tools in the list so it knows how
    # to use them.
    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        context=utils.split_model_and_provider(model),
    )
    return {"messages": [msg]}


async def store_memory(state: State, runtime: Runtime[Context]):
    # Extract tool calls from the last message
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    # Concurrently execute all upsert_memory calls
    saved_memories = await asyncio.gather(
        *(
            tools.upsert_memory(
                **tc["args"],
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store),
            )
            for tc in tool_calls
        )
    )

    # Format the results of memory storage operations
    # This provides confirmation to the model that the actions it took were completed
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        # If there are tool calls, we need to store memories
        return "store_memory"
    # Otherwise, finish; user can send the next message
    return END


# Create the graph + all nodes
builder = StateGraph(State, context_schema=Context)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response
builder.add_edge("store_memory", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"


__all__ = ["graph"]




File Name: prompts.py

-----------------------------

File Content: 

"""Define default prompts."""

SYSTEM_PROMPT = """You are a helpful and friendly chatbot. Get to know the user! \
Ask questions! Be spontaneous! 
{{user_info}}

System Time: {{time}}"""




File Name: state.py

-----------------------------

File Content: 

"""Define the shared values."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass(kw_only=True)
class State:
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


__all__ = [
    "State",
]




File Name: tools.py

-----------------------------

File Content: 

"""Define he agent's tools."""

import uuid
from typing import Annotated, Optional

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore


async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: Optional[uuid.UUID] = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"




File Name: utils.py

-----------------------------

File Content: 

"""Utility functions used in our graph."""


def split_model_and_provider(fully_specified_name: str) -> dict:
    """Initialize the configured chat model."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return {"model": model, "provider": provider}
'''