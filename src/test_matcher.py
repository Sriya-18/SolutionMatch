import src.data_loader as dl
from src.encoder import build_index, encode_texts
from src.matcher import find_top_matches

# Load data and build embeddings
df = dl.load_products()
prod_emb = build_index(df)

# Encode a sample query
query = "I need a compact gadget for home automation"
q_emb = encode_texts([query])[0]

# Find top 3 matches
matches = find_top_matches(q_emb, prod_emb, top_k=3)
print("Top matches (index, score):", matches)
print("Corresponding products:")
for idx, score in matches:
    print(df.iloc[idx]['name'], f"({score:.2f})")
