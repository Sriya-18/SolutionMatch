import pandas as pd

def load_products(path: str = "data/products.csv") -> pd.DataFrame:
    """
    Load the product catalog CSV into a pandas DataFrame.
    """
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df

if __name__ == "__main__":
    # Quick test
    df = load_products()
    print("Loaded products:")
    print(df.head())
