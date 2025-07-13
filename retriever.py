# retriever.py
import logging
import torch
from typing import List

from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.llms.llm import LLM
from prompts import get_query_generation_prompt
from transformers import pipeline

class HyPARetriever(BaseRetriever):
    def __init__(self,
                 llm: LLM,
                 vector_retriever: VectorIndexRetriever,
                 reranker: SentenceTransformerRerank,
                 param_mappings: dict,
                 rewriter: bool = True,
                 verbose: bool = False):
        self._llm = llm
        self._vector_retriever = vector_retriever
        self._reranker = reranker
        self._rewriter = rewriter
        self._verbose = verbose

        self.param_mappings = param_mappings
        super().__init__()

    def _classify_and_get_params(self, query: str) -> dict:
        # Ambil label dari custom_embedding_strs dalam QueryBundle
        label = 'LABEL_0'
        if hasattr(self, "_latest_query_bundle") and hasattr(self._latest_query_bundle, "custom_embedding_strs"):
            custom_labels = self._latest_query_bundle.custom_embedding_strs
            if custom_labels:
                label = custom_labels[0]
        if self._verbose: logging.info(f"Label parameter: {label}")
        return self.param_mappings.get(label, self.param_mappings.get('LABEL_0'))

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        self._latest_query_bundle = query_bundle  # Simpan untuk digunakan di _classify_and_get_params
        params = self._classify_and_get_params(query_bundle.query_str)
        top_k = params.get("k", 5)
        num_queries = params.get("Q", 3)

        if self._verbose: logging.info(f"Using dynamic params: top_k={top_k}, num_queries={num_queries}")

        queries = self._generate_queries(query_bundle.query_str, num_queries)
        retrieved_nodes = []
        for q in queries:
            retrieved_nodes.extend(self._vector_retriever.retrieve(q))

        unique_nodes = list({node.node.node_id: node for node in retrieved_nodes}.values())
        self._reranker.top_n = top_k
        reranked_nodes = self._reranker.postprocess_nodes(unique_nodes, query_bundle)
        return reranked_nodes
    
    def _generate_queries(self, query_str: str, num_queries: int) -> List[str]:
        if not self._rewriter or num_queries <= 1:
            return [query_str]
        prompt = get_query_generation_prompt(query_str, num_queries)
        response = self._llm.complete(prompt)
        queries = response.text.strip().split("\n")
        all_queries = [query_str] + [q.strip() for q in queries if q.strip()]
        if self._verbose: logging.info(f"Generated Queries: {all_queries}")
        return all_queries
