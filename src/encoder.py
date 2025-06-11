from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-MPNet-base-v2")

def encode_texts(texts):
    """
    Encode a list of strings into embeddings.
    Returns a NumPy array of shape (len(texts), embedding_dim).
    """
    # This may download ~400MB the first time you run it
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

def build_index(df, text_col="description"):
    """
    Given a DataFrame and the name of its text column,
    returns a 2D NumPy array of embeddings for that column.
    """
    texts = df[text_col].tolist()
    return encode_texts(texts)
