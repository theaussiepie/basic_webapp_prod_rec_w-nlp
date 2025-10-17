from flask import Flask, render_template, request
import pandas as pd
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)

# --- Load dataset ---
df = pd.read_csv("data/clothes.csv")  
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

stemmer = PorterStemmer()

# --- Create URL-safe slugs ---
def slugify(title):
    # Lowercase, remove non-alphanumeric characters except spaces and hyphens, replace spaces with hyphens
    title = title.lower()
    title = re.sub(r"[^\w\s-]", "", title)  # remove special characters
    title = re.sub(r"\s+", "-", title.strip())  # replace spaces with hyphens
    return title

df["slug"] = df["Clothes_Title"].apply(slugify)

# --- Routes ---

# Home page: show categories and unique clothing items
@app.route("/")
def home():
    categories = sorted(df["Class_Name"].dropna().unique())
    
    # Get search query from the URL parameters (GET request)
    query = request.args.get("q", "").strip().lower()
    
    # Start with unique items
    unique_items = df.drop_duplicates(subset=["Clothes_Title"])

    if query:
        # Stem the search query
        query_stem = stemmer.stem(query)

        # Create a mask: stem the item titles and class names
        mask = unique_items["Clothes_Title"].str.lower().apply(stemmer.stem).str.contains(query_stem) | \
               unique_items["Class_Name"].str.lower().apply(stemmer.stem).str.contains(query_stem)

        unique_items = unique_items[mask]

    return render_template(
        "index.html",
        clothes=unique_items.to_dict(orient="records"),
        categories=categories,
        query=query,
        result_count=len(unique_items)
    )

# Category page: show unique items in that category
@app.route("/category/<category>")
def category(category):
    filtered = df[df["Class_Name"] == category]
    categories = sorted(df["Class_Name"].dropna().unique())
    unique_items = filtered.drop_duplicates(subset=["Clothes_Title"])
    return render_template(
        "category.html",
        clothes=unique_items.to_dict(orient="records"),
        category=category,
        categories=categories
    )

# Item details page: show all reviews for selected item
@app.route("/item/<item_slug>")
def item_detail(item_slug):
    selected_item = df[df["slug"] == item_slug].drop_duplicates(subset=["Clothes_Title"])
    
    if selected_item.empty:
        return render_template("item_detail.html", item=None, reviews=[])

    item_reviews = df[df["slug"] == item_slug]

    return render_template(
        "item_detail.html",
        item=selected_item.iloc[0].to_dict(),
        reviews=item_reviews.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)