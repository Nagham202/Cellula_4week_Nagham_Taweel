import streamlit as st

from src.assistant import CodingAssistant
from src.code_runner import extract_python_code, run_code
from src.feedback import load_saved_records


st.set_page_config(page_title="AI Coding Assistant", page_icon="💻", layout="wide")

ROUTE_LABELS = {
    "explain": "🟢 Explain path (no RAG)",
    "generate": "🟡 Generate path (RAG + relevance check)",
    "not_relevant": "🔴 Not relevant: asking for a solution",
    "learn": "🔵 Learned a new solution",
}
# A big file would use up the per-minute token limit in a single message
MAX_FILE_CHARS = 6000


# Created once and reused on every rerun, because loading the model and Chroma takes a few seconds
@st.cache_resource
def load_assistant():
    return CodingAssistant()


def shorten(text, limit):
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def attach_uploaded_file(prompt):
    uploaded = st.session_state.get("uploaded_file")
    if uploaded is None:
        return prompt

    # Attach each uploaded file only once, not to every message after it
    file_key = f"{uploaded.name}-{uploaded.size}"
    if file_key in st.session_state.attached_files:
        return prompt
    st.session_state.attached_files.add(file_key)

    text = uploaded.getvalue().decode("utf-8", errors="replace")[:MAX_FILE_CHARS]
    return f"{prompt}\n\nAttached file `{uploaded.name}`:\n```python\n{text}\n```"


def show_run_result(run):
    if run["timed_out"]:
        st.warning(f"Stopped after {run['seconds']}s (time limit).")
    elif run["return_code"] == 0:
        st.success(f"Finished without errors in {run['seconds']}s")
    else:
        st.error(f"Finished with an error (return code {run['return_code']})")

    if run["stdout"]:
        st.caption("Output (stdout)")
        st.code(run["stdout"], language="text")
    if run["stderr"]:
        st.caption("Errors (stderr)")
        st.code(run["stderr"], language="text")


def show_assistant_message(index, message):
    st.markdown(message["content"])

    with st.expander(f"How I answered: {ROUTE_LABELS.get(message['route'], message['route'])}"):
        for step in message["trace"]:
            st.markdown(f"- {step}")

    code = extract_python_code(message["content"])
    if message["route"] == "generate" and code:
        # Every button needs its own key. The result is stored in the message so it stays visible after the page reruns
        if st.button("▶ Run code", key=f"run_{index}"):
            with st.spinner("Running in a separate process..."):
                message["run_result"] = run_code(code)
        if "run_result" in message:
            show_run_result(message["run_result"])


def show_sidebar(assistant):
    memory = assistant.memory.data

    with st.sidebar:
        st.header("🧠 Memory")
        preferences = memory["preferences"]
        column1, column2 = st.columns(2)
        column1.metric("Language", preferences["language"] or "—")
        column2.metric("Framework", preferences["framework"] or "—")
        st.caption(f"{len(memory['turns'])} turns saved · {len(memory['generated_code'])} code answers")

        if assistant.pending_request:
            st.info(f"Waiting for your solution to:\n\n*{assistant.pending_request}*\n\n"
                    "Paste it in the chat and I will learn it.")

        with st.expander("Previous questions"):
            if not memory["turns"]:
                st.caption("Nothing yet.")
            for turn in reversed(memory["turns"][-10:]):
                st.markdown(f"- `{turn['route']}` {shorten(turn['user'], 70)}")

        with st.expander("Generated code"):
            if not memory["generated_code"]:
                st.caption("Nothing yet.")
            for item in reversed(memory["generated_code"][-5:]):
                st.caption(shorten(item["request"], 70))
                st.code(item["code"], language="python")

        with st.expander("What the LLM sees from memory"):
            st.code(assistant.memory.as_prompt_text(), language="markdown")

        st.header("📚 Knowledge base")
        # _collection is Chroma's own object; count() gives the number of stored chunks
        st.caption(f"{assistant.vector_store._collection.count()} chunks in Chroma · "
                   f"{len(load_saved_records())} taught by users")

        st.header("📎 Attach a file")
        st.file_uploader("A .py or .txt file is added to your next message",
                         type=["py", "txt"], key="uploaded_file")

        st.divider()
        column1, column2 = st.columns(2)
        if column1.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
        if column2.button("Clear memory"):
            assistant.memory.clear()
            assistant.pending_request = None
            st.rerun()


try:
    assistant = load_assistant()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

# session_state survives reruns (Streamlit reruns this whole file on every click)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "attached_files" not in st.session_state:
    st.session_state.attached_files = set()

show_sidebar(assistant)

st.title("💻 AI Coding Assistant")
st.caption("Ask me to **explain** code or to **write** code. I check my knowledge base before writing, "
           "and if I don't know something, you can teach me.")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_assistant_message(index, message)

prompt = st.chat_input("e.g. Explain this code: ...  /  Write a function that ...")
if prompt:
    prompt = attach_uploaded_file(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Classifying, searching and judging happen here, then the answer streams in piece by piece
            with st.status("Thinking...", expanded=False) as status:
                result = assistant.handle(prompt)
                status.update(label=ROUTE_LABELS[result["route"]], state="complete")
            answer = st.write_stream(result["answer_stream"])
        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "route": result["route"],
        "trace": result["trace"],
    })
    st.rerun()
