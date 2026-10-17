from langchain_core.prompts import ChatPromptTemplate

from src.config import MAX_FOLLOW_UP_CHARS
from src.prompts import STANDALONE_HUMAN, STANDALONE_SYSTEM

STANDALONE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STANDALONE_SYSTEM),
    ("human", STANDALONE_HUMAN),
])


def make_standalone_request(message, memory, llm):
    # Nothing to fill in on the first message, and long messages carry their own context
    if not memory.data["turns"] or len(message) > MAX_FOLLOW_UP_CHARS:
        return message

    messages = STANDALONE_PROMPT.format_messages(
        message=message,
        memory=memory.as_prompt_text(include_code=False),
    )
    rewritten = llm.invoke(messages).content.strip()
    return rewritten or message
