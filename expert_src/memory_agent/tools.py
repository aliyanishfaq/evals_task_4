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
    response = interrupt(f"Saving the following memory: {content} in the category: {category}. Please reply with 'accept' or 'reject'")
    if response == "accept":
        pass
    else:
        return f"Rejected memory: {content} in the category: {category}"

    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id, category),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"
