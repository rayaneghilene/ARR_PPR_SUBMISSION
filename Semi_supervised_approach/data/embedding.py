from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

def embed_texts(texts: List[str], model_name: str = 'all-mpnet-base-v2', batch_size: int = 64, show_progress_bar: bool = True) -> np.ndarray:
    """Encode a list of texts into sentence embeddings (numpy array)."""
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
    return np.asarray(embs, dtype=np.float32)