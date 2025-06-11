import src.data_loader as dl
from src.encoder import build_index

# Load products
df = dl.load_products("data/products.csv")
# Build embeddings
embeddings = build_index(df)
print(f"Generated embeddings of shape {embeddings.shape}")
