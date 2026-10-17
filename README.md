# AI Coding Assistant (Cellula NLP Internship, Week 4)

This week I built a coding assistant you can chat with in Streamlit. You can paste code and ask what it does, or ask it to write something new. The more interesting part is what happens in between. It decides what kind of request you sent, it checks whether it actually knows enough before writing code, and when it doesn't, it asks you to teach it. It also remembers the conversation, so follow-ups like "now make it ignore negative numbers" work.

## What happens when you send a message

1. **Is this the answer to a question I asked?** If the assistant just said "I don't know this one, can you show me?", it first checks whether your message is that solution. If it is, it saves it to its knowledge base and thanks you.
2. **Explain or generate?** An LLM labels the message as *Code Explanation* or *Code Generation*.
3. **Explain path.** The LLM explains the code directly. It doesn't search anything or use embeddings, because the answer is in the code you pasted.
4. **Generate path.** This one has more steps:
   - If the message is a short follow-up like "in python", it's rewritten into a full request using the conversation memory.
   - The top 3 most similar solved problems are pulled from Chroma.
   - A second LLM call acts as a judge and answers *Relevant* or *Not Relevant*.
   - If relevant, the assistant writes complete code with comments, type hints and error handling.
   - If not, it doesn't guess. It says: *"I couldn't find relevant knowledge for your request. Could you provide the correct solution or reference so I can learn it for future interactions?"*
5. **Run it.** Under generated code there's a **▶ Run code** button. The code runs in a separate process, never inside the LLM, and you see what it printed and any errors.

Every answer has a small "How I answered" section that shows the route it took, which documents it found and what the judge decided. It made debugging much easier, and it's nice for demos.

```
message ─▶ (waiting for a solution? ─▶ save it to Chroma)
        ─▶ classifier ─┬─ Explain ──▶ LLM explains (no RAG)
                       └─ Generate ─▶ rewrite follow-up ─▶ Chroma top-3 ─▶ judge ─┬─ Relevant ─▶ LLM writes code ─▶ ▶ Run
                                                                                  └─ Not Relevant ─▶ ask the user to teach it
memory (preferences, recent turns, last code) is passed to every LLM call
```

## What I used and why

- **LLM: Groq, `openai/gpt-oss-120b`, temperature 0.** I first planned `llama-3.3-70b-versatile`, but my account returned "model not found", so I listed the models my key can actually use and picked this one. It's fast and much better at code than the 8B models I can run locally. One setting in `src/config.py` switches to Ollama (`llama3.1:8b`) if I need to work offline.
- **Embeddings: `nomic-embed-text` through Ollama.** I reused my Week 3 setup, including the `search_document:` / `search_query:` prefixes. Requests are written in English, like the problem descriptions, so a text embedding model works well here.
- **Knowledge base: HumanEval.** It has 164 Python problems. I kept 20 out of Chroma (fixed random seed) so I could later test whether the assistant really writes correct code, instead of copying a stored answer. Each problem is one document (the function signature, its docstring and the official solution), and it isn't split. Splitting would separate the description from the code. The longest problem is under 2,000 characters anyway.
- **Chunking.** The splitter is still there, and it's Python-aware: it cuts at `def` and `class` lines. It matters for long solutions that users teach the assistant.
- **Vector store: Chroma, saved on disk.** It uses cosine distance.
- **Memory: everything is saved to `data/memory.json`.** But only a small recent part goes to the LLM: my language and framework preferences, the last 3 turns and the last generated code. Sending the full history every time would hit the token limits quickly.

## How I tested it

| What | Result |
|---|---|
| Code runner: all 164 official HumanEval solutions against their own tests | **164/164 pass** |
| Intent classifier: 20 messages, 10 per label, incl. 2 in Arabic | **20/20** |
| Relevance judge: 17 requests (list/string problems vs. Flask, PyTorch, databases, GUIs…) | **17/17** |
| Manual end-to-end runs in the app | explain ✔, generate + run ✔, CNN → not relevant → I taught it → asked again → relevant ✔, follow-ups with memory ✔ |

I also wrote `evaluation/eval_pass_at_1.py` (pass@1 on the 20 held-out problems) and `evaluation/run_scenarios.py` (4 scenarios with automatic checks). I haven't run them yet, because I reached Groq's daily token limit.

To be honest about the numbers: the test sets are small, and a few classifier test messages are close to the examples in the prompt, so 100% is probably a bit optimistic.

## Things that went wrong (and what I learned)

- **An English question answered in Japanese.** My prompt said "answer in the same language as the user". But the message was mostly code, so the model had to guess. Now the rule is explicit: Arabic gets Arabic, everything else gets English.
- **The judge wasn't sure about "sum of the even numbers in a list".** It said Not Relevant 5 times in a row, even though very similar problems were found. Reading the model's reasoning showed it couldn't decide what "relevant" meant. I added one deciding question ("does the request need knowledge the context doesn't show?") plus a few examples, and it became stable.
- **The assistant learned something wrong.** I typed "in python" as a follow-up. The search looked for just those two words, found nothing, and the assistant asked me to teach it. My next message was a *question* that contained code, and it got saved as the "solution". I fixed both causes: short follow-ups are rewritten using memory before searching, and the feedback check now knows that a message asking something is never a solution. I also added a way to undo bad learning: `python -m scripts.forget_solution <doc_id>`. Testing in the real app found this bug. Single-message tests never would have.
- **Rate limits.** The free tier gives 8,000 tokens per minute and 200,000 per day per model. The client now waits and retries instead of crashing, but a busy minute still makes answers slow.

## Running it

You need Python 3.12 and [Ollama](https://ollama.com).

```bash
ollama pull nomic-embed-text
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                     # put your Groq API key in .env
python -m scripts.build_knowledge_base   # loads HumanEval into Chroma (~5 s)
streamlit run app.py
```

Every file in `src/` has a small self-test, so you can check pieces one at a time, for example `python -m src.code_runner` or `python -m src.retriever`.

Evaluation scripts (run them one at a time because of the token limits):

```bash
python -m scripts.check_canonical_solutions
python -m evaluation.eval_classifier
python -m evaluation.eval_relevance
python -m evaluation.eval_pass_at_1        # add --limit 3 for a quick try
python -m evaluation.run_scenarios
```

## Project layout

```
app.py                  the Streamlit UI (no logic, just display)
src/
  assistant.py          the router: connects everything below
  classifier.py         explain or generate?
  query_rewriter.py     "in python" -> a full request
  retriever.py          top-k search in Chroma
  relevance_judge.py    Relevant / Not Relevant
  explainer.py          explain path (no RAG)
  generator.py          writes the code
  feedback.py           detects, saves and forgets user-taught solutions
  memory.py             long-term memory
  code_runner.py        runs code safely
  knowledge_base.py     HumanEval -> Chroma
  embeddings.py, llm.py, prompts.py, config.py
scripts/                build the knowledge base, check solutions, forget a solution
evaluation/             test sets, scripts and saved results
data/                   Chroma, held-out problems, taught solutions, memory
```

## About the code runner's safety

Code only runs when you click the button. It runs in its own Python process, with a 10-second limit, inside a fresh temporary folder. It runs with a clean environment, so it **can't see the API key** (I checked: it prints `None`). It can't read keyboard input, and its output is cut off if it gets huge. This is not a full sandbox, though. The code can still read files on the machine and use the internet. For a local tool where I read the code before running it, that's acceptable. Running it in Docker would be the proper next step.

## Known limitations

- Free-tier token limits make some answers slow, and they can stop the app for the rest of the day.
- Preferences are detected by keywords, so "don't use Flask" still records Flask as a preference.
- The runner only has the libraries installed in `.venv`. PyTorch or Flask code shows an import error, and a web server keeps running until the time limit stops it.
- Removing a wrongly learned solution is a manual step.
