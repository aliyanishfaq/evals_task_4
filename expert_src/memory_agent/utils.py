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
