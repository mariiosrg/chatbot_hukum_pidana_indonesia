from typing import List

def get_query_generation_prompt(query_str: str, num_queries: int) -> str:
    """
    Menghasilkan prompt untuk membuat beberapa sub-pertanyaan dari pertanyaan asli.
    """
    return (
        f"Anda adalah seorang ahli dalam menyaring pertanyaan pengguna menjadi {num_queries} sub-pertanyaan "
        f"yang dapat digunakan untuk menjawab pertanyaan asli secara lengkap: '{query_str}'.\n"
        f"Keluarkan sub-pertanyaan yang ditulis ulang, satu di setiap baris, tanpa teks lain apa pun."
    )