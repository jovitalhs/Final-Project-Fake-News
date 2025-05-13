import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import pickle
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ——— Load trained model (Pipeline includes TF-IDF + Classifier) ———
with open('best_fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ——— HTML Header ———
html_temp = """
<div style="background-color:#000;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Fake News Prediction App</h1> 
    <h4 style="color:#fff;text-align:center">Detect whether a news article is REAL or FAKE</h4>
</div>
"""

# ——— Description ———
desc_temp = """ 
### About This App
This app uses a TF-IDF + LinearSVC model to classify news as *REAL* or *FAKE* based on its title and content.

#### Data & Code Source
Kaggle: [Fake News Prediction (92.5% Accuracy)](https://www.kaggle.com/code/rajatkumar30/fake-news-prediction-92-5-accuracy/input)
"""

# ——— Text cleaning function ———
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# ——— Streamlit UI ———
def main():
    stc.html(html_temp, height=120)
    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown(desc_temp, unsafe_allow_html=True)

    elif choice == "Prediction":
        st.subheader("Enter News Title and Content")
        title = st.text_input("News Title")
        content = st.text_area("News Content", height=200)

        if st.button("Predict"):
            if not title or not content:
                st.error("Please provide both title and content.")
            else:
                combined = f"{title} {content}"
                cleaned = clean_text(combined)
                pred = model.predict([cleaned])[0]
                label = "REAL 📰" if pred == 1 else "FAKE 🚩"
                st.success(f"Prediction: *{label}*")

if __name__ == "__main__":
    main()

    