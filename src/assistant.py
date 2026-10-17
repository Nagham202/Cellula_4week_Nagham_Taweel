import time

from src.classifier import EXPLAIN, classify_intent
from src.code_runner import extract_python_code, run_code
from src.explainer import stream_explanation
from src.feedback import is_solution, save_solution
from src.generator import stream_generation
from src.knowledge_base import load_vector_store
from src.llm import get_llm
from src.memory import ConversationMemory
from src.query_rewriter import make_standalone_request
from src.relevance_judge import RELEVANT, judge_relevance
from src.retriever import retrieve

NOT_RELEVANT_MESSAGE = (
    "I couldn't find relevant knowledge for your request. "
    "Could you provide the correct solution or reference so I can learn it for future interactions?"
)


# Wraps a fixed reply so it looks like a stream, just like the LLM answers
def single_message(text):
    yield text


class CodingAssistant:
    def __init__(self):
        self.llm = get_llm()
        self.vector_store = load_vector_store()
        self.pending_request = None
        self.memory = ConversationMemory()

    def handle(self, message):
        result = self.route(message)
        # The full answer only exists after the last piece is streamed, so memory is saved at the end
        result["answer_stream"] = self.remember_when_done(message, result["route"], result["answer_stream"])
        return result

    def remember_when_done(self, message, route, answer_stream):
        full_answer = ""
        for piece in answer_stream:
            full_answer += piece
            yield piece
        self.memory.add_turn(message, route, full_answer)

    def route(self, message):
        trace = []

        # If I just asked the user to teach me, check whether this message is the answer before classifying it
        if self.pending_request:
            learned = self.feedback_path(message, trace)
            if learned:
                return learned

        short_memory = self.memory.as_prompt_text(include_code=False)
        intent, raw_answer = classify_intent(message, self.llm, short_memory)
        trace.append(f"Classifier -> {intent}")

        if intent == EXPLAIN:
            return self.explain_path(message, intent, trace)
        return self.generate_path(message, intent, trace)

    def feedback_path(self, message, trace):
        request = self.pending_request
        looks_like_solution, raw_answer = is_solution(request, message, self.llm)

        self.pending_request = None

        if not looks_like_solution:
            trace.append("Feedback check -> New Request (not saved)")
            return None

        trace.append("Feedback check -> Solution")
        doc_id, chunk_count = save_solution(request, message, self.vector_store)
        trace.append(f"Embedded + metadata + inserted into Chroma as {doc_id} ({chunk_count} chunk)")

        reply = (f"Thank you! I saved your solution to my knowledge base as `{doc_id}`.\n\n"
                 f"Next time someone asks for something like \"{request}\", I can use it.")
        return {"intent": "Feedback", "route": "learn", "trace": trace,
                "sources": [], "answer_stream": single_message(reply)}

    def explain_path(self, message, intent, trace):
        trace.append("Explain path: LLM only (no retrieval)")
        answer_stream = stream_explanation(message, self.llm, self.memory.as_prompt_text())
        return {"intent": intent, "route": "explain", "trace": trace,
                "sources": [], "answer_stream": answer_stream}

    def generate_path(self, message, intent, trace):
        # Something like "in python" means nothing on its own, so turn it into a full request using memory
        request = make_standalone_request(message, self.memory, self.llm)
        if request != message:
            trace.append(f"Follow-up rewritten as: {request}")

        results = retrieve(request, self.vector_store)
        documents = [document for document, distance in results]
        found = ", ".join(f"{doc.metadata['doc_id']} ({distance:.2f})" for doc, distance in results)
        trace.append(f"Retrieved top-{len(results)}: {found}")

        memory_text = self.memory.as_prompt_text()
        # Don't trust the search blindly: Chroma always returns something, even when nothing fits
        verdict, raw_answer = judge_relevance(request, documents, self.llm, memory_text)
        trace.append(f"Relevance judge -> {verdict}")

        if verdict == RELEVANT:
            trace.append("LLM code generation with the retrieved context")
            answer_stream = stream_generation(message, documents, self.llm, memory_text)
            return {"intent": intent, "route": "generate", "trace": trace,
                    "sources": results, "answer_stream": answer_stream}

        # Remember what we couldn't answer, so the next message can be saved as its solution
        self.pending_request = request
        trace.append("Not relevant -> asking the user for a solution")
        return {"intent": intent, "route": "not_relevant", "trace": trace,
                "sources": results, "answer_stream": single_message(NOT_RELEVANT_MESSAGE)}


if __name__ == "__main__":
    assistant = CodingAssistant()
    assistant.memory.clear()

    conversation = [
        "In Python, write a function that returns the sum of the even numbers in a list",
        "Now make it ignore any values that are not integers",
        "Explain the code you just wrote, briefly",
        "Create a CNN model in PyTorch for image classification",
    ]

    for message in conversation:
        print("=" * 70)
        print("USER:", message.splitlines()[0])
        start = time.time()
        result = assistant.handle(message)
        answer = "".join(result["answer_stream"])
        print("TRACE:\n  " + "\n  ".join(result["trace"]))
        print(f"ASSISTANT ({result['route']}, {time.time() - start:.1f}s):")
        print(answer[:700] + ("\n..." if len(answer) > 700 else ""))

    print("=" * 70)
    print("SAVED MEMORY:", len(assistant.memory.data["turns"]), "turns,",
          len(assistant.memory.data["generated_code"]), "generated code,",
          "preferences =", assistant.memory.data["preferences"])
