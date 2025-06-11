import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_top_matches(query_embedding: np.ndarray,
                     product_embeddings: np.ndarray,
                     top_k: int = 5):
    """
    Given a single query embedding (1×D) and a matrix of product embeddings (N×D),
    returns a list of (index, similarity_score) for the top_k most similar products.
    If there are zero products, returns an empty list.
    """
    # EARLY EXIT: no products to compare
    if product_embeddings.shape[0] == 0:
        return []

    # Compute cosine similarities
    sims = cosine_similarity(
        query_embedding.reshape(1, -1),
        product_embeddings
    )[0]

    # Get the indices of the top_k highest scores
    best_idxs = np.argsort(sims)[-top_k:][::-1]
    return [(int(idx), float(sims[idx])) for idx in best_idxs]
