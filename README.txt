EDA was completed iteratively and congruently alongside webapp.

Training data for reviews were synthetically added to match the kaggle dataset:
 https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews




How to run the web app:
1. Create a virtual environment
2. Activate the virtual environment
3. Install dependencies
   pip install -r requirements.txt
4. Download NLTK stopwords (only once):
   python
   >>> import nltk
   >>> nltk.download('stopwords')
   >>> exit()
5. Run the Flask app:
   python app.py
6. Open a browser 




