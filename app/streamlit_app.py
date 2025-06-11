import sys, os
# 1) Make sure your src folder is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

import csv
import streamlit as st
import pandas as pd
from data_loader import load_products
from encoder import encode_texts, build_index
from matcher import find_top_matches

# 2) Page config must come first
st.set_page_config(page_title="SolutionMatch", layout="wide")

# 3) App title & description
st.title("SolutionMatch: NLP-Driven Product Matching")
st.markdown("""
Enter a description of what you need and see the top product matches based on semantic similarity!
""")

# 4) Load & cache data + embeddings
@st.cache_data
def init():
    df = load_products("data/products.csv")
    embeddings = build_index(df)
    return df, embeddings

df_original, embeddings_original = init()

# 5) Sidebar filters
st.sidebar.header("Filters")
categories = ["All"] + sorted(df_original["category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", categories)
min_price, max_price = st.sidebar.slider(
    "Price range",
    float(df_original.price.min()), float(df_original.price.max()),
    (float(df_original.price.min()), float(df_original.price.max()))
)

# 6) Apply filters
df = df_original.copy()
embeddings = embeddings_original.copy()

if selected_cat != "All":
    mask = df["category"] == selected_cat
    df = df[mask]
    embeddings = embeddings[mask.values]

price_mask = (df.price >= min_price) & (df.price <= max_price)
df = df[price_mask]
embeddings = embeddings[price_mask.values]

# 7) Main query input
query = st.text_input("🔍 Describe your requirements here:", "")

def log_query(query_text, matches):
    os.makedirs("data", exist_ok=True)
    with open("data/query_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([query_text] + [idx for idx, _ in matches])

# 8) Matching logic with early exit
if st.button("Find Matches") and query:
    # 8a: If no products remain, show error and stop
    if embeddings.shape[0] == 0:
        st.error("❗ No products match your filters. Please adjust filters and try again.")
        st.stop()

    # 8b: Otherwise proceed
    with st.spinner("Finding best matches…"):
        q_emb = encode_texts([query])[0]
        try:
            matches = find_top_matches(q_emb, embeddings, top_k=5)
        except Exception as e:
            st.error(f"Error computing matches: {e}")
            st.stop()

    # 8c: Display results or warning
    if matches:
        st.subheader("Top Matches")
        for idx, score in matches:
            prod = df.iloc[idx]
            # If you have images, uncomment:
            # st.image(prod["image_url"], width=150)
            st.markdown(f"**{prod['name']}** — _Score: {score:.2f}_")
            st.write(prod["description"])
            st.write(f"Category: {prod['category']}  •  Price: ${prod['price']:.2f}")
            st.markdown("---")
        log_query(query, matches)
    else:
        st.warning("No matches found. Try a different query.")

