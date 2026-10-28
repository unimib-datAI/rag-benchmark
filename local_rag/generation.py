"""Answer generation with an OpenAI-compatible chat model (e.g. Azure AI Foundry)."""
from functools import cache

from openai import OpenAI

import config

SYSTEM_PROMPT = (
    "Answer the question using only the documents below. "
    "If they do not contain the answer, say that you don't know. "
    "Answer concisely, in the language of the question."
)


@cache
def client():
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def answer(question, documents):
    context = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(documents, start=1))
    options = {"temperature": float(config.LLM_TEMPERATURE)} if config.LLM_TEMPERATURE else {}
    response = client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"},
        ],
        **options,
    )
    return response.choices[0].message.content
