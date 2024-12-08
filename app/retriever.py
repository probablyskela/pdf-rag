from rank_bm25 import BM25Okapi
import numpy as np
from utils import chunk_document

from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, docs: list[str]) -> None:
        self.docs = [list(chunk_document(doc)) for doc in docs]

        tokenized_docs = [doc.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(corpus=tokenized_docs)
        self.sbert = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

        self.doc_embeddings = self.sbert.encode(docs)

    def get_docs(
        self,
        query: str,
        n: int = 3,
        keywords: bool = True,
        semantic: bool = True,
    ) -> list[str]:
        zeros = np.zeros(len(self.docs))
        bm25_scores = 0.3 * self._get_bm25_scores(query=query) if keywords else zeros
        semantic_scores = 0.7 * self._get_semantic_scores(query=query) if semantic else zeros

        scores = bm25_scores + semantic_scores

        sorted_indices = np.argsort(scores)[::-1]

        return [self.docs[i] for i in sorted_indices[:n]]

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        tokenized_query = query.lower().split()

        return self.bm25.get_scores(tokenized_query)

    def _get_semantic_scores(self, query: str) -> np.ndarray:
        query_embedding = self.sbert.encode(query)

        return np.array(self.sbert.similarity(np.array(query_embedding), self.doc_embeddings)[0])
