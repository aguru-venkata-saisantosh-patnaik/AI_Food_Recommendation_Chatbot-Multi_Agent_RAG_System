import psutil
from langchain_huggingface import HuggingFaceEmbeddings


def setup_embeddings_cpu():
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings

    # Setup embeddings
