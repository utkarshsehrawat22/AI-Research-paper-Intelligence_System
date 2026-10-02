"""Optional Groq-powered RAG assistant over the local FAISS paper index."""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


class ResearchRAGAssistant:
    """Retrieve papers locally, then generate grounded answers with Groq."""

    def __init__(self, engine):
        self.engine = engine
        api_key = os.getenv("GROQ_API_KEY")
        self.available = bool(api_key)
        self.llm = None
        if self.available:
            self.llm = ChatGroq(
                model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                temperature=0.2,
                api_key=api_key,
            )

    def route(self, query: str) -> str:
        lowered = query.lower()
        if any(word in lowered for word in ("compare", "comparison", "difference", "versus", " vs ")):
            return "compare"
        if any(word in lowered for word in ("keyword", "keywords", "topics", "concepts")):
            return "keywords"
        return "search"

    def _context(self, results: list[dict]) -> str:
        blocks = []
        for number, result in enumerate(results, start=1):
            blocks.append(
                f"[Paper {number}]\nTitle: {result['title']}\n"
                f"Similarity: {result['score']:.3f}\n"
                f"Abstract: {result['abstract'][:3500]}"
            )
        return "\n\n".join(blocks)

    def answer(self, query: str, k: int = 5) -> dict:
        if not self.available:
            raise RuntimeError("GROQ_API_KEY is not configured. Add it to a local .env file.")

        results = self.engine.search(query, k=max(k, 3))
        mode = self.route(query)
        instruction = {
            "search": "Answer the research question using the retrieved abstracts.",
            "keywords": "Identify and explain the most important research topics in the retrieved papers.",
            "compare": "Compare the retrieved papers. Discuss their problem, method, strengths, limitations, and key differences.",
        }[mode]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a careful research assistant. Use only the supplied paper context. "
             "If the context is insufficient, say so. Do not invent citations or results. "
             "{instruction}"),
            ("human", "Question: {query}\n\nPaper context:\n{context}"),
        ])
        response = (prompt | self.llm).invoke({
            "instruction": instruction,
            "query": query,
            "context": self._context(results),
        })
        return {"mode": mode, "answer": response.content, "sources": results}
