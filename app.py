import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Set the NLTK data path to the local directory
nltk.data.path.append('./nltk_data')

# Set Streamlit theme to light
st.set_page_config(layout="centered", page_title="Emotion Detection", page_icon="🙂")

# Header
st.title("Emotion Detection in Text")

# CSS to hide the "Press Enter to apply" placeholder text in the text input
hide_placeholder_style = """
    <style>
    .stTextInput input::placeholder {
        color: transparent;
    }
    .stTextInput div[role="alert"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_placeholder_style, unsafe_allow_html=True)

# Specify the correct path to your dataset
file_path = 'data.csv'

# 1. Explore the Data
@st.cache_data(show_spinner=False)  # Cache the data loading to improve performance
def load_data(file_path, encodings):
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return None

data = load_data(file_path, ['ISO-8859-1', 'latin1', 'cp1252'])

if data is None:
    st.write("Failed to load the data. Please check the file path and encoding.")
else:
    # Stack 'Farxad' and 'Murug' vertically, with labels
    data_farxad = data[['Farxad']].dropna().rename(columns={'Farxad': 'text'})
    data_farxad['label'] = 'Farxad'

    data_murug = data[['Murug']].dropna().rename(columns={'Murug': 'text'})
    data_murug['label'] = 'Murug'

    # Combine into one DataFrame
    data_combined = pd.concat([data_farxad, data_murug])

    # Display the counts for each class to verify the distribution
    st.write("Class Distribution")
    class_distribution = data_combined['label'].value_counts()
    st.bar_chart(class_distribution)

    ## Create visualizations
    # Generate word clouds
    @st.cache_data(show_spinner=False)  # Cache the word cloud generation
    def generate_wordclouds(data_combined):
        wordclouds = {}
        for label in ['Farxad', 'Murug']:
            subset = data_combined[data_combined['label'] == label]
            text = " ".join(review for review in subset.text)
            wordcloud = WordCloud(background_color="white").generate(text)
            wordclouds[label] = wordcloud.to_array()
        return wordclouds

    wordclouds = generate_wordclouds(data_combined)

    # Word Cloud
    for label, image in wordclouds.items():
        st.write(f"Word Cloud for {label}")
        st.image(image)

    ## Perform basic preprocessing
    @st.cache_data(show_spinner=False)  # Cache the function to improve performance
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    data_combined['processed_text'] = data_combined['text'].apply(preprocess)

    @st.cache_data(show_spinner=False)  # Cache the data preparation to improve performance
    def prepare_data(data_combined):
        tfidf_vect = TfidfVectorizer()
        tfidf = tfidf_vect.fit_transform(data_combined['processed_text'])
        X_train, X_test, y_train, y_test = train_test_split(tfidf, data_combined['label'], test_size=0.1, random_state=42)
        return X_train, X_test, y_train, y_test, tfidf_vect

    X_train, X_test, y_train, y_test, tfidf_vect = prepare_data(data_combined)

    @st.cache_data(show_spinner=False)  # Cache the model training to improve performance
    def train_models(_X_train, _y_train):
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(_X_train, _y_train)

        rf = RandomForestClassifier()
        rf.fit(_X_train, _y_train)

        svm = SVC()
        svm.fit(_X_train, _y_train)

        mlp = MLPClassifier(max_iter=1000)
        mlp.fit(_X_train, _y_train)

        return logreg, rf, svm, mlp

    logreg, rf, svm, mlp = train_models(X_train, y_train)

    # Evaluate each model
    y_pred_logreg = logreg.predict(X_test)
    st.write("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_logreg))

    y_pred_rf = rf.predict(X_test)
    st.write("Random Forest accuracy:", accuracy_score(y_test, y_pred_rf))

    y_pred_svm = svm.predict(X_test)
    st.write("SVM accuracy:", accuracy_score(y_test, y_pred_svm))

    y_pred_mlp = mlp.predict(X_test)
    st.write("MLP Classifier accuracy:", accuracy_score(y_test, y_pred_mlp))

    ## Function to preprocess and predict a new statement
    def predict_new_statement(statement, model, vectorizer):
        preprocessed_statement = preprocess(statement)
        transformed_statement = vectorizer.transform([preprocessed_statement])
        prediction = model.predict(transformed_statement)
        return prediction[0]

    ## Input a new statement and predict its category
    new_statement = st.text_input("Soo gali oraah:")

    if st.button("Submit"):
        if new_statement:
            # Check if any word in the new statement exists in the data
            all_words = set(" ".join(data_combined['text']).split())
            new_words = set(new_statement.split())
            if new_words.intersection(all_words):
                predicted_label = predict_new_statement(new_statement, logreg, tfidf_vect)
                st.write(f"oraahda la soo galiyay waxey ka turjumeysaa : {predicted_label}")
            else:
                st.write("This word is not known.")

