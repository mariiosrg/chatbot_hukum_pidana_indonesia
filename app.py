import streamlit as st
import weaviate
import torch
import logging
import os
from dotenv import load_dotenv

# LlamaIndex & Model Imports
from llama_index.core import VectorStoreIndex, Settings, QueryBundle
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

# --- PERUBAHAN 1: Impor kelas-kelas baru Anda ---
from qa_classifier import QAClassifier
from retriever import HyPARetriever
from llm_components import JinaEmbedding, TogetherLLM
from weaviate import connect_to_wcs
from weaviate.auth import AuthApiKey

import nest_asyncio

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
st.set_page_config(page_title="Chatbot Hukum Pidana (HyPA-RAG)", layout="wide")

# Tentukan path atau nama model classifier Anda di Hugging Face Hub
# Contoh path lokal: "./my_indonesian_q_classifier"
# Contoh nama di HF: "mariiosrg/nama-model-anda"
CLASSIFIER_MODEL_ID = "mariiosrg/IndoLegalBERT" 

try:
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not all([WEAVIATE_URL, WEAVIATE_API_KEY, TOGETHER_API_KEY]):
        raise ValueError("Satu atau lebih API key tidak ditemukan di file .env")
except (ValueError, KeyError) as e:
    st.error(f"Error: Gagal memuat API keys. Pastikan file .env sudah diatur. Detail: {e}")
    st.stop()

# --- FUNGSI INISIALISASI ---
@st.cache_resource
def setup_base_components():
    try:
        import weaviate
        from weaviate.auth import AuthApiKey

        client = weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY),
            skip_init_checks=True  # ⬅️ solusi untuk melewati gRPC health check
        )

        if not client.is_ready():
            raise ConnectionError("❌ Klien Weaviate tidak siap atau gagal koneksi ke WCS.")

        Settings.embed_model = JinaEmbedding()
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name="Hukum",
            text_key="content",
            metadata_keys=["source", "pasal"]
        )
        vector_index = VectorStoreIndex.from_vector_store(vector_store)

        logging.info("✅ Komponen dasar berhasil diinisialisasi.")
        return vector_index

    except Exception as e:
        st.error(f"❌ Gagal saat inisialisasi komponen dasar: {e}")
        st.stop()

# --- PERUBAHAN 2: Fungsi baru untuk memuat classifier dengan caching ---
@st.cache_resource
def load_qa_classifier(model_id: str):
    """Memuat model klasifikasi QA dari path lokal atau Hugging Face Hub."""
    try:
        return QAClassifier(model_id=model_id)
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        st.warning("Fitur deteksi kompleksitas otomatis dinonaktifkan.")
        return None

def setup_chat_engine(vector_index, system_prompt, llm_params, rewriter_enabled, source_filter: str):
    # Fungsi ini tidak lagi berhubungan dengan classifier, menjadi lebih sederhana
    llm = TogetherLLM(**llm_params)
    filters = None
    if source_filter and source_filter != "Semua Dokumen":
        filters = MetadataFilters(filters=[MetadataFilter(key="source", value=source_filter)])
    
    vector_retriever = vector_index.as_retriever(similarity_top_k=8, filters=filters)
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-large", top_n=5)
    
    # Retriever tidak lagi memerlukan classifier_model, karena klasifikasi terjadi di luar
    param_mappings_2_class = {
        'LABEL_0': {"k": 3, "Q": 1},   # Simple: cukup 1 query dan ambil 3 dokumen
        'LABEL_1': {"k": 8, "Q": 2},  # Complex: butuh 2 reformulasi dan ambil 8 dokumen
    }

    hypa_retriever = HyPARetriever(
        llm=llm,
        vector_retriever=vector_retriever,
        reranker=reranker,
        param_mappings=param_mappings_2_class,
        rewriter=rewriter_enabled,
        verbose=True
    )
    
    chat_history = st.session_state.get("chat_history", [])
    memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history[-3:])
    return ContextChatEngine.from_defaults(retriever=hypa_retriever, memory=memory, llm=llm, system_prompt=system_prompt)
import re

def extract_answer(text: str) -> str:
    """
    Membersihkan bagian <think>...</think> atau variasinya dari output LLM,
    lalu menghapus karakter '▌' sisa dari streaming.
    """
    # Hapus semua tag <think> (case-insensitive, bisa ada spasi)
    clean_text = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Hapus karakter '▌' dan whitespace sisa
    clean_text = clean_text.replace("▌", "").strip()

    return clean_text

# --- UI DAN LOGIKA CHAT ---
st.title("⚖️ Chatbot Konsultasi Hukum Pidana (HyPA-RAG)")

with st.sidebar:
    st.image("law-auction-svgrepo-com.svg", width=150)
    st.header("⚙️ Pengaturan")
    legal_sources_options = ["Semua Dokumen", "KUHP", "KUHAP", "UU-Narkotika", "UU-ITE"]
    source_filter = st.selectbox("Batasi pencarian pada sumber hukum:", options=legal_sources_options)
    rewriter_enabled = st.checkbox("Aktifkan Query Rewriter", value=True)
    st.markdown("---")
    st.info("Kompleksitas pertanyaan kini dideteksi secara otomatis oleh AI.")
    system_prompt = "Anda adalah asisten hukum pidana Indonesia yang memberikan jawaban akurat dan ringkas berdasarkan KUHP, KUHAP, dan UU Pidana lainnya. Gunakan selalu bahasa Indonesia. "
    temperature = 0.3
    max_tokens = 1200
    top_p = 0.9
    st.info("Disclaimer: Chatbot ini adalah prototipe dan bukan pengganti nasihat hukum profesional.")

# --- PERUBAHAN 3: Memuat semua komponen di awal ---
vector_index = setup_base_components()
qa_classifier = load_qa_classifier(CLASSIFIER_MODEL_ID)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ada yang bisa saya bantu terkait hukum pidana Indonesia?"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan tentang KUHP, Narkotika, ITE..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Menganalisis & mencari referensi hukum... ⏳"):
            
            # --- PERUBAHAN 4: Logika klasifikasi dijalankan di sini menggunakan objek QAClassifier ---
            classification_label = "LABEL_0" # Default jika classifier gagal
            if qa_classifier:
                classification_label = qa_classifier.predict(prompt)
            
            # Buat QueryBundle untuk "membawa" label ke retriever
            query_bundle = QueryBundle(
                query_str=prompt,
                custom_embedding_strs=[classification_label]
            )

            # Inisialisasi chat engine
            llm_params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
            chat_engine = setup_chat_engine(vector_index, system_prompt, llm_params, rewriter_enabled, source_filter)
            
            # Lakukan streaming chat dengan query bundle yang sudah dimodifikasi
            streaming_response = chat_engine.stream_chat(query_bundle)

            response_container = st.empty()
            expander_container = st.empty()

            full_response = ""
            visible_response = ""
            thinking_buffer = ""
            inside_think = False

            for token in streaming_response.response_gen:
                full_response += token

                # Deteksi awal dan akhir tag <think>
                if "<think>" in token.lower():
                    inside_think = True
                    continue  # skip penampilan langsung

                if "</think>" in token.lower():
                    inside_think = False
                    continue  # skip penampilan langsung

                if inside_think:
                    thinking_buffer += token
                else:
                    visible_response += token
                    response_container.markdown(visible_response + "▌")

            # Tampilkan final jawaban (tanpa '▌')
            final_response = visible_response.strip("▌").strip()
            cleaned_response = extract_answer(final_response)

            # Tampilkan konten <think> di expander (jika ada)
            if thinking_buffer.strip():
                with expander_container.expander("Tampilkan proses berpikir 🧠"):
                    st.markdown(thinking_buffer.strip())

            # Tampilkan ulang isi yang sudah bersih (ganti respons lama)
            response_container.markdown(cleaned_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.chat_history.extend([ChatMessage(role=MessageRole.USER, content=prompt), ChatMessage(role=MessageRole.ASSISTANT, content=full_response)])
            
            if streaming_response.source_nodes:
                with st.expander("Lihat Sumber Referensi 📚"):
                    for i, node in enumerate(streaming_response.source_nodes):
                        properties = node.node.metadata.get("properties", {}) if node.node and node.node.metadata else {}

                        source_doc = properties.get("source", "Tidak diketahui")
                        pasal_doc = properties.get("pasal", "Tidak diketahui")

                        try:
                            content_doc = node.node.get_content().strip()
                        except Exception:
                            content_doc = "Konten tidak tersedia"

                        st.markdown("---")
                        st.markdown(f"##### 📖 Sumber {i+1}")
                        st.markdown(f"**Relevansi Skor:** `{node.score:.2f}`")
                        st.markdown(f"**Sumber Dokumen:** `{source_doc}`")
                        st.markdown(f"**Pasal Terkait:** `{pasal_doc}`")
                        st.info(f"**Isi Konten:**\n\n> {content_doc}")




