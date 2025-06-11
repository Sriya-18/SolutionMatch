import sys, os
# Ensure src/ is on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

import csv
import streamlit as st
import pandas as pd
from data_loader import load_products
from encoder import encode_texts, build_index
from matcher import find_top_matches

# 1. Page config (must be the first Streamlit call)
st.set_page_config(page_title="SolutionMatch", layout="wide")

# 2. Title and description
st.title("SolutionMatch: NLP-Driven Product Matching")
st.markdown("""
Enter a description of what you need and see the top product matches based on semantic similarity!
""")

# 3. Load data & embeddings (cached for performance)
@st.cache_data
def init():
    df = load_products("data/products.csv")
    embeddings = build_index(df)
    return df, embeddings

df_original, embeddings_original = init()

# 4. Sidebar filters
st.sidebar.header("Filters")

# 4a. Category filter
categories = ["All"] + sorted(df_original["category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", categories)

# 4b. Price slider
min_price, max_price = st.sidebar.slider(
    "Price range",
    float(df_original.price.min()),
    float(df_original.price.max()),
    (float(df_original.price.min()), float(df_original.price.max()))
)

# Apply filters to data and embeddings
df = df_original.copy()
embeddings = embeddings_original.copy()

if selected_cat != "All":
    mask = df["category"] == selected_cat
    df = df[mask]
    embeddings = embeddings[mask.values]

price_mask = (df.price >= min_price) & (df.price <= max_price)
df = df[price_mask]
embeddings = embeddings[price_mask.values]

# 5. Main input area
query = st.text_input("🔍 Describe your requirements here:", "")

# 6. Query logging function
def log_query(query_text, matches):
    """Append query and matched indices to a log CSV."""
    os.makedirs("data", exist_ok=True)
    with open("data/query_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([query_text] + [idx for idx, _ in matches])

# 7. Matching & display logic
if st.button("Find Matches") and query:
    with st.spinner("Finding best matches…"):
        q_emb = encode_texts([query])[0]
        matches = find_top_matches(q_emb, embeddings, top_k=5)

    st.subheader("Top Matches")
    for idx, score in matches:
        prod = df.iloc[idx]
        # Uncomment if you have an image_url column:
        # st.image(prod["image_url"], width=150)
        st.markdown(f"**{prod['name']}** — _Score: {score:.2f}_")
        st.write(prod["description"])
        st.write(f"Category: {prod['category']}  •  Price: ${prod['price']:.2f}")
        st.markdown("---")

    # 8. Log that query and its matches
    log_query(query, matches)







