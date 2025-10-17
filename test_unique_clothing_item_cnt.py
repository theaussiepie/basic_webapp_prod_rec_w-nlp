import pandas as pd

# Load dataset
df = pd.read_csv("data/clothes.csv")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Count total rows
total_rows = len(df)

# Count unique clothing items
unique_items = df["Clothes_Title"].nunique()

# List unique titles (optional)
unique_titles = df["Clothes_Title"].unique()

print(f"Total rows in CSV: {total_rows}")
print(f"Number of unique clothing items: {unique_items}")
print("Unique clothing titles:")
for title in unique_titles:
    print("-", title)