"""Define default prompts."""

SYSTEM_PROMPT = """You are a helpful and friendly chatbot. Get to know the user! \
Ask questions! Be spontaneous! 
{user_info}

System Time: {time}"""


CATEGORY_PROMPT = """
Based on these recent messages: {messages}
    
Which memory categories are relevant? Choose from: personal, professional, other
    
    Respond with only the category name.
    Examples:
    - "professional" (for work discussions)
    - "personal" (for personal topics)  
    - "other" (for other topics)

Your response should always be one word from personal, professional, or other. 
"""