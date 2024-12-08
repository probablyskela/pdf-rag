from groq import Groq
from retriever import Retriever

PROMPT = """\
You are a helpful assistant that can answer questions.

Rules:
- Reply with the answer only.
- Say 'I don't know' if you don't know the answer.
- Use the provided context.
"""


class RAG:
    def __init__(self, api_key: str, docs: list[str]) -> None:
        self.client = Groq(api_key=api_key)
        self.retriever = Retriever(docs=docs)

    def ping(self) -> bool:
        try:
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model="llama3-8b-8192",
            )
        except Exception:
            return False

        return True

    def answer_question(self, question: str, keywords: bool, semantic: bool) -> str | None:
        context = self.retriever.get_docs(
            query=question,
            n=10,
            keywords=keywords,
            semantic=semantic,
        )

        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Context:\n{context}\nQuestion:\n{question}\n"},
            ],
            model="llama3-8b-8192",
            stream=False,
        )

        return completion.choices[0].message.content
