from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.prompts import CLASSIFIER_HUMAN, CLASSIFIER_SYSTEM

EXPLAIN = "Code Explanation"
GENERATE = "Code Generation"

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CLASSIFIER_SYSTEM),
    ("human", CLASSIFIER_HUMAN),
])


def parse_label(raw_answer):
    text = raw_answer.strip().lower()
    # Only look for the key word, so "Code Explanation." or "label: code explanation" still count
    if "explanation" in text:
        return EXPLAIN
    if "generation" in text:
        return GENERATE
    return None


def classify_intent(message, llm, memory_text="Conversation memory: no previous conversation."):
    messages = CLASSIFIER_PROMPT.format_messages(message=message, memory=memory_text)
    raw_answer = llm.invoke(messages).content
    label = parse_label(raw_answer)

    # Never crash on a strange answer. Generation is the safer guess because it checks the knowledge base first
    if label is None:
        print(f"[classifier] could not parse {raw_answer!r}, using {GENERATE}")
        label = GENERATE
    return label, raw_answer


if __name__ == "__main__":
    llm = get_llm()
    examples = [
        "Explain this code:\nnums = [x * x for x in range(5)]",
        "Write a function that checks if a number is prime",
    ]
    for example in examples:
        label, raw_answer = classify_intent(example, llm)
        print(f"{label:17s} (raw: {raw_answer!r})  <- {example.splitlines()[0]}")
