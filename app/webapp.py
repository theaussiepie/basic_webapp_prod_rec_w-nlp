#import necessary libraries
from flask import Flask, render_template, request, redirect, url_for 
import pandas as pd
from nltk.stem import PorterStemmer
import re
from joblib import load
import os
import string
from nltk.corpus import stopwords

# text pre-processing
stop_words = set(stopwords.words("english"))
token_pattern = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def preprocess_text(text):
    text = text.lower()
    tokens = token_pattern.findall(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

# load pre-trained models and vectoriser
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
tfidf_vectorizer = load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
logreg_model = load(os.path.join(MODEL_DIR, 'logreg_tfidf_model.pkl'))

# initialise Flask app
app = Flask(__name__) 

# load dataset
df = pd.read_csv("data/clothes.csv")  
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

#stemmer for item search functionality
stemmer = PorterStemmer()

# slug creation for URLs
def slugify(title):
    # Lowercase, remove non-alphanumeric characters except spaces and hyphens,
    # replace spaces with hyphens
    title = title.lower()
    title = re.sub(r"[^\w\s-]", "", title)  # remove special characters
    title = re.sub(r"\s+", "-", title.strip())  # replace spaces with hyphens
    return title

df["slug"] = df["Clothes_Title"].apply(slugify)


# temporary in-memory storage for reviews (these are removed on restart)
temp_reviews = {}




# Routes

# Home page: show categories and unique clothing items
@app.route("/")
def home():
    #from df, drop missing values, make unique, sort
    categories = sorted(df["Class_Name"].dropna().unique())
    
    # read search keyword from user input, lowercase, strip
    query = request.args.get("q", "").strip().lower()
    
    # start with unique items
    unique_items = df.drop_duplicates(subset=["Clothes_Title"])

    if query:
        # use Porterstemmer to stem query
        query_stem = stemmer.stem(query)

        # search filter logic. result boolean
        mask = unique_items["Clothes_Title"].str.lower().apply(stemmer.stem).str.contains(query_stem) | \
               unique_items["Class_Name"].str.lower().apply(stemmer.stem).str.contains(query_stem)

        # keep values where mask == True
        unique_items = unique_items[mask]

    # display results dynamically, show count, keep search bar filled
    return render_template(
        "index.html",
        clothes=unique_items.to_dict(orient="records"),
        categories=categories,
        query=query,
        result_count=len(unique_items)
    )

# category page: show unique items in that category
@app.route("/category/<category>")

# shows items filtered by category
def category(category):
    filtered = df[df["Class_Name"] == category]
    categories = sorted(df["Class_Name"].dropna().unique())
    unique_items = filtered.drop_duplicates(subset=["Clothes_Title"])

    # open category.html with filtered items
    return render_template(
        "category.html",
        clothes=unique_items.to_dict(orient="records"),
        category=category,
        categories=categories
    )

# item details page: show all reviews for selected item
@app.route("/item/<item_slug>")
def item_detail(item_slug):
    # find item that matches slug
    selected_item = df[df["slug"] == item_slug].drop_duplicates(subset=["Clothes_Title"])
    if selected_item.empty:
        return render_template("item_detail.html", item=None, reviews=[])

    #  get all original dataset reviews
    item_reviews = df[df["slug"] == item_slug].to_dict(orient="records")

    # normalize column name for template matching
    for r in item_reviews:
        if "Recommended_IND" in r:
            r["Recommended"] = r["Recommended_IND"]

    # append temporary reviews if made
    if item_slug in temp_reviews:
        item_reviews.extend(temp_reviews[item_slug])

    # load item.html, passes item and its reviews
    return render_template(
        "item_detail.html",
        item=selected_item.iloc[0].to_dict(),
        reviews=item_reviews
    )

# add review page. get and post methods
# item slug route tied to specific item
@app.route("/item/<item_slug>/add_review", methods=["GET", "POST"])

# add review function
def add_review(item_slug):
    # match df to slug
    selected_item = df[df["slug"] == item_slug].drop_duplicates(subset=["Clothes_Title"])
    # if no match, 404
    if selected_item.empty:
        return "Item not found", 404

    #if user submitted form
    if request.method == "POST":
        # user typed the review but not confirmed
        if "confirm" not in request.form:
            # read user input into html form
            title = request.form["title"]
            review = request.form["review"]
            rating = request.form["rating"]

            # join title and review for prediction
            combined_text = f"{title} {review}"
            clean_text = preprocess_text(combined_text) #pre-process
            transformed = tfidf_vectorizer.transform([clean_text]) #vectorise clean text

            # debug: show what text is actually used for prediction
            # first 200 characters to print to terminal
            print(f"[DEBUG] Cleaned input text: {clean_text[:200]}...")

            #  predict using the Logistic Regression model
            predicted_label = int(logreg_model.predict(transformed)[0]) # 0 or 1
            proba = logreg_model.predict_proba(transformed)[0] # prob for both classes
            # debug: show prediction result to terminal
            print(f"[DEBUG] Predicted label: {predicted_label}, Probabilities: {proba}")

            # DEBUG ! : show top contributions
            feature_names = tfidf_vectorizer.get_feature_names_out()
            coefs = logreg_model.coef_[0]
            present_words = transformed.nonzero()[1]

            # sort by contribution (coefficient × TF-IDF weight)
            word_contrib = [(feature_names[i], coefs[i] * transformed[0, i]) for i in present_words]
            word_contrib.sort(key=lambda x: x[1], reverse=True)

            top_positive = word_contrib[:5]
            top_negative = word_contrib[-5:]

            #print top 5 pos and ne contributors to terminal
            print(f"[DEBUG] Top positive contributors: {top_positive}")
            print(f"[DEBUG] Top negative contributors: {top_negative}")

            # renders new page with prediction results, option to confirm or overrise
            return render_template(
                "review_result.html",
                item=selected_item.iloc[0].to_dict(),
                review={"Title": title, "Review_Text": review, "Rating": rating, "Recommended": predicted_label},
            )

        # second submission: user confirmed (can override or not)
        title = request.form["title"]
        review = request.form["review"]
        rating = request.form["rating"]
        predicted_label = int(request.form["predicted_label"])
        override = request.form.get("override")

        #apply override if user selected
        if override is not None:
            predicted_label = int(override)

        #save new review temporarily
        new_review = {
            "Title": title,
            "Review_Text": review,
            "Rating": rating,
            "Recommended": predicted_label,
        }
        temp_reviews.setdefault(item_slug, []).append(new_review)

        # return back to item_detail page
        return redirect(url_for("item_detail", item_slug=item_slug))

    #user can see add_review.html form empty
    return render_template("add_review.html", item=selected_item.iloc[0].to_dict())






if __name__ == "__main__":
    app.run(debug=True)