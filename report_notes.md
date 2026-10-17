# Week 4 – AI Coding Assistant: Report Notes

## Step 0 – Big picture and setup

### System overview
Every message goes: feedback check → intent classifier (LLM) → router →
- **Explain path:** LLM explains the code directly (no RAG, no embeddings).
- **Generate path:** retrieve top-k from Chroma → relevance judge (LLM) →
  "Relevant": generate code / "Not Relevant": ask the user for the solution.
- A user-provided solution is embedded and added to Chroma (feedback learning).
- Memory (past questions, past code, preferences) is passed to every LLM call.
- Code runs only through a separate code runner tool, when the user clicks Run.

### Decisions
| Decision | Choice | Why |
|---|---|---|
| File organization | Thin `app.py` (UI only) + one job per file in `src/` | Each piece can be tested alone; evaluation scripts reuse them without the UI |
| Router | Plain Python `if/else` (no LangGraph) | Only two routes; easier to read and explain |
| Build order | Code runner moved from step 8 to step 3 | It has no LLM dependency; lets us verify HumanEval's official tests early and run generated code in later steps |
| Environment | `.venv` with Python 3.12, same as Week 3 | Isolated libraries per project |

### Final build order
0 Setup → 1 LLM → 2 Knowledge base (HumanEval) → 3 Code runner → 4 Classifier →
5 Router + explain → 6 Generate path → 7 Feedback learning → 8 Memory →
9 Streamlit → 10 Testing, README, report

### Open issue (solved in Step 6)
Held-out HumanEval problems are not in Chroma, so a strict judge would say "Not Relevant"
and pass@1 could never generate code. → Solved by defining "Relevant" as "same language and
same kind of problem", not "the exact same problem" (see Step 6).

### Challenges
-

## Step 1 – LLM
- Choice: **Groq `openai/gpt-oss-120b`** (free API) as default; Ollama `llama3.1:8b` as offline fallback.
- One switch `LLM_PROVIDER` in `src/config.py`; all code calls `get_llm()` from `src/llm.py`.
- Why Groq: a 120B model is much better than 8B at code and at following "answer with one word" rules; very fast, so evaluations finish in minutes.
- Risk: free-tier rate limits / needs internet → switch the one setting to "ollama".
- API key in `.env` (git-ignored), loaded with python-dotenv; `.env.example` shows the format.
- Same model for classifier, judge and generator for now (temperature 0).
- Result: invoke 0.49s, streaming works.
- Challenge: `llama-3.3-70b-versatile` gave 404 "model not found" although the docs listed it → listed the models my key can use with `models.list()` and switched to `gpt-oss-120b`. Lesson: check what the account can access, not just the docs.

## Step 2 – Knowledge base (HumanEval → Chroma)
- Loaded 164 problems with Hugging Face `datasets`; shuffled with seed 42 → **144 in Chroma, 20 held out** (`data/held_out_tasks.json`, with official tests, for pass@1).
- One document = `prompt + canonical_solution` (a complete function) + a header line `# Python solution: <entry_point> (<task_id>)`.
- Metadata: `doc_id`, `source="humaneval"`, `task_id`, `entry_point`, `language`. Chroma id = `<doc_id>#<chunk number>` → no duplicates.
- Chunking: **kept problems whole** (longest = 1993 chars, nomic limit 2048 tokens ≈ 6000+ chars); splitting would separate the docstring from its code. Used a **Python-aware splitter** (`from_language(Language.PYTHON)`, size 2500 / overlap 200) so long user-taught solutions are cut at `def`/`class` boundaries.
- Embeddings: reused Week 3 `nomic-embed-text` with search_document/search_query prefixes (already installed, queries are natural language like the docstrings).
- Search test (cosine distance, lower = closer): palindrome → `is_palindrome` 0.151; prime factor → `largest_prime_factor` 0.151; **Flask API 0.38–0.41, CNN 0.38–0.40** → unrelated requests still return k results, only further away. This is why we need the relevance judge.
- Challenge: chunk size 2000 split the longest problem (1993 + header) into 2 chunks → raised to 2500.

## Step 3 – Code runner tool
- `run_code(code)` in `src/code_runner.py`: code string → `main.py` in a fresh temp folder → separate Python process → returns stdout, stderr, return code, timed_out, seconds.
- Safety layers: runs only on button click; separate process; 10 s timeout; temp working folder (deleted after); **clean environment** (code cannot read `GROQ_API_KEY`); isolated mode `-I`; no stdin; output cut at 10,000 chars.
- Not a full sandbox: code can still read my files and use the network. Docker would fix that but needs setup → future work. Acceptable because only I run it, locally, after reading the code.
- `extract_python_code()` pulls ```python blocks out of the LLM's Markdown answer (skips ```bash).
- `run_humaneval_tests()` = solution + official `check()` + `check(entry_point)`.
- Results: self-test OK (normal, error→stderr, infinite loop stopped at 3 s, API key = None, runs in temp folder). **164/164 canonical solutions passed in 4.8 s** → the runner and test harness are correct.

## Step 4 – Intent classifier
- `classify_intent(message, llm)` in `src/classifier.py`; prompt in `src/prompts.py`.
- Reliability: clear definitions of both labels + 6 short examples (few-shot) + "reply with exactly one label" + temperature 0 + **forgiving parser** (looks for the word "explanation"/"generation", ignores case/punctuation) + **fallback to Code Generation** if unparseable (the app never crashes; generation is the safer path because it checks the knowledge base and asks the user).
- Definition choice: rewriting/converting code ("convert this to a list comprehension") = Generation (output is new code); "why is this wrong / is there a bug" = Explanation (PDF example).
- Test set: 20 messages (10/10), incl. 2 Arabic, a bug question, a complexity question, a "convert" request.
- **Result: 20/20 = 100%, ~0.41 s per message** (`evaluation/results_classifier.json`).
- Limitation: 3 test messages are close to the prompt's own examples (PDF examples), so 100% is likely optimistic; the set is small. Follow-ups without context ("now make it faster") are handled later with memory.

## Step 5 – Router + explain path
- `CodingAssistant` class in `src/assistant.py` = the router. Loads the LLM and Chroma **once**, then `handle(message)` → classifier → plain `if/else` → path.
- Every path returns the same shape: `{intent, route, trace, answer_stream}` → the UI treats all paths the same way.
- `trace` records the steps (e.g. "Classifier -> Code Explanation | Explain path: LLM only") → shows the routing in the UI and proves the explain path skips retrieval.
- Explain path (`src/explainer.py`): one LLM call, **streamed** with a Python generator (`yield`). It imports nothing from retriever/knowledge_base/embeddings → no RAG by design.
- Explainer prompt: summary → step by step → bug cause + fix → don't invent problems → answer in the user's language.
- Result: infinite-loop question → correct explanation + fix (`i += 1`); first words after 0.72 s, total 1.4 s.
- Challenge: one run answered an English question **in Japanese**. Cause: the rule "answer in the same language as the user's message" was vague (the message is mostly code) and temperature 0 is not fully deterministic on a reasoning model served by an API. Fix: explicit rule "look only at the user's words; Arabic → Arabic, otherwise English, never another language". Re-test: 6/6 English, 2/2 Arabic correct.

## Step 6 – Generate path (retrieval → relevance judge → generation)
- Flow in `CodingAssistant.generate_path()`: retrieve top-3 from Chroma → `judge_relevance()` → Relevant: `stream_generation()` / Not Relevant: fixed message from the task + remember `pending_request` (for Step 7).
- **Judge definition (key decision):** Relevant = same language AND same kind of problem (similar task, data types, techniques), not necessarily the exact problem. Not Relevant = needs a framework/library/language/domain the context doesn't show (Flask, PyTorch, databases, GUIs...).
  - Why not "exact same problem": held-out problems and any new list/string task would be Not Relevant → the assistant would be useless for the very problems it has good references for.
  - Why not a distance threshold: the task asks for an LLM judge; also distances overlap (relevant up to 0.29, unrelated from 0.355).
- Judge reliability: one-word answer, temperature 0, parser checks **"not relevant" before "relevant"** (substring trap), unparseable → **Not Relevant** (asking is safer than hallucinating).
- Generator prompt: one ```python block, type hints, docstring, WHY-comments, error handling **that must not change the described behaviour/edge cases**, keep given function signature (needed for HumanEval tests), `if __name__ == "__main__"` demo with prints (so the Run button shows output), no `input()`, standard library only unless needed, 2–4 bullet explanation, same language rule.
- **Relevance test: 16/16 = 100%** (8 Relevant incl. 2 held-out problems, 8 Not Relevant: Flask, CNN, PostgreSQL, Tkinter, BeautifulSoup, React, scikit-learn, SMTP) → `evaluation/results_relevance.json`.
- End-to-end: palindrome request → retrieved HumanEval/48, 10, 107 → Relevant → complete code → code runner: return code 0, 7 correct outputs. Flask → Not Relevant → asks the user.
- Challenge: some answers took 13–19 s. Timing showed classify 0.4 s, retrieve 0.05 s, but judge up to 6.6 s. Cause: **Groq free tier = 8,000 tokens/minute and 1,000 requests/day** (read from the response headers); running the 16-case evaluation used the minute's budget, so the client waited and retried. Normal chatting is fast; evaluations must be paced. Fallback: `LLM_PROVIDER = "ollama"`.

## Step 7 – Feedback learning
- After "Not Relevant", `pending_request` is remembered. The **next** message first goes through a **feedback check** (before the classifier).
- Detection decision: an LLM call answers "Solution" / "New Request" (given the pending request + new message). Rejected alternatives: "next message is always the solution" (breaks when the user moves on), "contains a code block" (misses text/link answers, catches new code questions), a special command (users forget it). Unsure → not saved (a wrong "solution" would pollute the knowledge base).
- Saving (`src/feedback.py`): document text = `# User-taught solution for: <request>` + the user's message (so similar future requests find it) → metadata `doc_id`, `source="user_feedback"`, `request`, `added_at` → chunk + embed + insert with the same `add_documents()` as HumanEval.
- `doc_id = user/<sha1 of request+solution>` → teaching the same thing twice replaces instead of duplicating.
- Also appended to `data/user_solutions.jsonl`; `build_knowledge_base` adds them back after a rebuild (otherwise rebuilding would erase everything learned). Verified: 145 chunks after rebuild, Flask doc still found.
- Scenario result: Flask → Not Relevant → user teaches → saved as `user/7b068a1f46` → "Create a Flask API that has a /users route" → retrieved `user/7b068a1f46` at **0.22** (HumanEval at 0.40) → judge Relevant → Flask code generated. CNN → Not Relevant → user asks an unrelated explain question → "New Request", nothing saved, explained normally.
- Limitation: running a Flask server with the Run button will hit the 10 s timeout (servers never stop) and Flask is not installed in the runner → expected.

## Step 8 – Memory
- `ConversationMemory` in `src/memory.py`, saved to `data/memory.json` (survives restarts = long-term; git-ignored, personal data).
- Stores: every turn (time, user message, route, answer), every generated code (request + code), preferences (language, framework).
- Design decision: **save everything on disk, but send only a small recent part to the LLM**: preferences + last 3 turns (400 chars each) + last generated code (2000 chars). Rejected: full history (grows without limit; Groq free tier is 8,000 tokens/min) and LLM summaries (an extra LLM call every message).
- Preferences: **keyword detection** (whole-word regex, so "java" ≠ "javascript"); latest mention wins. Free and transparent; limitation: "don't use Flask" would still set Flask. LLM extraction would be smarter but costs a call per message.
- Memory goes into **all four prompts**: classifier (short version, no code), explainer ("the code you just wrote"), judge (follow-ups on remembered code count as Relevant), generator (follow preferences, modify previous code). The explain path still has no retrieval: memory is conversation history, not RAG.
- Saving: the answer stream is wrapped by a generator (`remember_when_done`) that saves the turn after the last piece → the UI does not need to remember to save.
- Scenario result: sum of evens → code; "now make it ignore values that are not integers" → modified the same function (`isinstance` check added); "explain the code you just wrote" → explained that exact code; CNN in PyTorch → still **Not Relevant** (memory did not fool the judge). Saved: 4 turns, 2 code, preferences = Python / PyTorch.
- Challenge 1: **429 rate-limit crash** (8,000 tokens/min) → `max_retries=6` in `get_llm()` so the client waits and retries. Turns now take 12–36 s during bursts (waiting, not computing).
- Challenge 2: judge said **Not Relevant 5/5** for "sum of the even numbers in a list" (closest 0.17). The model's reasoning showed it was unsure ("none directly sum evens… could be considered relevant?"). Fix: added examples + the explicit deciding question "does the request need knowledge the context does not show?" → 5/5 Relevant.
- Relevance test updated: Flask is now *correctly* Relevant (taught in Step 7) → replaced by Django; added the borderline even-sum case. **17/17 = 100%.**

## Step 9 – Streamlit app (`app.py`)
- `app.py` is UI only; all logic stays in `src/` (`CodingAssistant` loaded once with `st.cache_resource`).
- Chat: `st.chat_message` + `st.chat_input`; history in `st.session_state.messages` (each assistant message keeps its route, trace and run result).
- Streaming: classifier/retrieval/judge run inside `st.status("Thinking...")`, then `st.write_stream()` shows the answer piece by piece.
- Code blocks: `st.markdown` renders ```python with syntax highlighting + copy button.
- **Run button** only under generated code (route = generate): `extract_python_code()` → `run_code()` → stdout/stderr + success/error/timeout box. Result is stored inside the message so it survives Streamlit reruns; each button has a unique key.
- "How I answered" expander under every answer shows the trace (route, retrieved docs + distances, judge verdict).
- Sidebar (memory visualization, optional): language/framework, turn and code counts, pending request ("Waiting for your solution to…"), previous questions, generated code, the exact memory text the LLM sees, knowledge base size (chunks + user-taught), Clear chat / Clear memory.
- File upload (optional): .py/.txt added once to the next message (max 6,000 chars because of the token limit).
- Errors (Ollama off, no internet, Groq limit) shown with `st.error` instead of crashing.
- Test: headless `streamlit.testing` AppTest → no exceptions; factorial request → generate → Run → return code 0, output shown.
- **Bug found in manual UI testing:** user wrote the follow-up "in python" → Chroma was searched for just "in python" → Not Relevant → `pending_request = "in python"` → the user's next message (a *question* about code) was judged a "Solution" and **saved into the knowledge base** (`user/3cee411318`).
  - Root cause 1: memory reached the prompts but **not the search query**. Fix: `src/query_rewriter.py` rewrites short follow-ups (≤200 chars, only when there is memory) into a standalone request ("in python" → "Write a Python function that returns only the odd numbers from a given list.") used for retrieval, the judge and `pending_request`. The generator still gets the original message + memory.
  - Root cause 2: the feedback check was fooled by code in a question. Fix: prompt rule "a message that ASKS something is a New Request even if it contains code" + 4 examples.
  - Undo tool: `forget_solution()` + `python -m scripts.forget_solution <doc_id>` removes a wrong document from Chroma and the .jsonl; used it to delete `user/3cee411318` (backup kept).
  - Replay of the same scenario: "in python" → rewritten → Relevant → code; Tkinter → Not Relevant; the question → "New Request (not saved)" → explained; saved solutions 2 → 2.
  - Lesson: testing with a real conversation finds bugs that one-message tests don't; a self-learning knowledge base needs a way to undo bad learning.

## Step 10 – Testing, README
- Challenge: hit Groq's **daily** limit (200,000 tokens/day per model) for `gpt-oss-120b` after a day of development (429 "tokens per day"). Limits are per model; `gpt-oss-20b` / `qwen3.8-27b` have their own budgets. Decision on which model for which job: **deferred**.
- `evaluation/eval_pass_at_1.py`: 20 held-out problems → request "Complete this Python function…" + prompt → retrieve → judge → generate → `extract_python_code` → official tests with the code runner. Not Relevant / no code count as failures. Also checks that no held-out problem is retrieved (leak check). Checked without tokens: harness passes the canonical solution.
- `evaluation/run_scenarios.py`: 4 scenarios / 8 turns with automatic checks (route, trace contents, code runs); each scenario gets a fresh temporary memory; what scenario 3 teaches is removed afterwards so the test is repeatable.
- README.md: architecture, setup, run, evaluation commands, structure, runner safety, limitations.
- pass@1 and the scenario script are written but not run yet (Groq daily token limit). The 4 scenarios were tested by hand in the app instead.
