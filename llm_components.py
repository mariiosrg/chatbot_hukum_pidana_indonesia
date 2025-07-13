import os
import torch
from typing import List, Any, Generator

try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    def cache_decorator(func):
        return func

from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
from together import Together
from dotenv import load_dotenv
from pydantic import PrivateAttr

from llama_index.core.llms.llm import LLM
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole, CompletionResponse, LLMMetadata

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

@cache_decorator
def load_embedding_model() -> SentenceTransformer:
    print("Memuat model embedding Jina...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device=device)

class JinaEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._model = load_embedding_model()
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query, normalize_embeddings=True).tolist()
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

class TogetherLLM(CustomLLM):
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 0.9
    _client: Any = PrivateAttr()
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = Together(api_key=TOGETHER_API_KEY)
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name, is_chat_model=True, context_window=4096, num_output=self.max_tokens)
    def _get_model_kwargs(self, **kwargs: Any) -> dict:
        base_kwargs = {"temperature": self.temperature, "max_tokens": self.max_tokens, "top_p": self.top_p}
        return {**base_kwargs, **kwargs}
    def _chat_to_dict(self, messages: List[ChatMessage]) -> List[dict]:
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        model_kwargs = self._get_model_kwargs(**kwargs)
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(model=self.model_name, messages=messages, stream=False, **model_kwargs)
        return CompletionResponse(text=response.choices[0].message.content, raw=response.dict())
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        model_kwargs = self._get_model_kwargs(**kwargs)
        api_messages = self._chat_to_dict(messages)
        response_stream = self._client.chat.completions.create(model=self.model_name, messages=api_messages, stream=True, **model_kwargs)
        def gen() -> Generator[ChatResponse, None, None]:
            content_so_far = ""
            for chunk in response_stream:
                delta = chunk.choices[0].delta
                delta_content = delta.content or ""
                content_so_far += delta_content
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content_so_far), delta=delta_content, raw=chunk.dict())
        return gen()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        model_kwargs = self._get_model_kwargs(**kwargs)
        messages = [{"role": "user", "content": prompt}]
        response_stream = self._client.chat.completions.create(model=self.model_name, messages=messages, stream=True, **model_kwargs)
        def gen() -> Generator[CompletionResponse, None, None]:
            text_so_far = ""
            for chunk in response_stream:
                delta = chunk.choices[0].delta
                delta_content = delta.content or ""
                text_so_far += delta_content
                yield CompletionResponse(text=text_so_far, delta=delta_content, raw=chunk.dict())
        return gen()