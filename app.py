# Import the libraries 
import streamlit as st 
import pickle 
import numpy as np 
import pandas as pd 
import os

st.set_page_config(layout="wide")

st.header("Book Recommender System")

st.markdown('''
##### The site using collaborative filtering suggests books from our catalog.  
##### We recommend top 50 books for everyone as well. 
''')

# Load pickled models/data safely
def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"File not found: {path}")
        return None

# Load all required pickle files
popular = load_pickle('popular.pkl')
book_data = load_pickle('books.pkl')  # renamed to avoid conflict with CSV
pt = load_pickle('pt.pkl')
similarity_scores = load_pickle('similarity_scores.pkl')

# Sidebar section for Top 50 books
st.sidebar.title("Top 50 Books")

if st.sidebar.button("SHOW"):
    if popular is not None:
        cols_per_row = 5 
        num_rows = 10 
        for row in range(num_rows): 
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row): 
                book_idx = row * cols_per_row + col
                if book_idx < len(popular):
                    with cols[col]:
                        st.image(popular.iloc[book_idx]['Image-URL-M'])
                        st.text(popular.iloc[book_idx]['Book-Title'])
                        st.text(popular.iloc[book_idx]['Book-Author'])

# Recommendation function
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    for i in similar_items:
        item = []
        temp_df = book_data[book_data['Book-Title'] == pt.index[i[0]]]
        temp_df = temp_df.drop_duplicates('Book-Title')
        item.extend(temp_df['Book-Title'].values)
        item.extend(temp_df['Book-Author'].values)
        item.extend(temp_df['Image-URL-M'].values)
        data.append(item)
    return data

# Dropdown for selecting books
if pt is not None:
    book_list = pt.index.values
    st.sidebar.title("Similar Book Suggestions")
    selected_book = st.sidebar.selectbox("Select a book from the dropdown", book_list)

    if st.sidebar.button("Recommend Me"):
        if book_data is not None:
            book_recommend = recommend(selected_book)
            cols = st.columns(5)
            for col_idx in range(5):
                with cols[col_idx]:
                    if col_idx < len(book_recommend):
                        st.image(book_recommend[col_idx][2])
                        st.text(book_recommend[col_idx][0])
                        st.text(book_recommend[col_idx][1])

# Load and show CSV data (used to train the model)
try:
    books_csv = pd.read_csv('Data/Books.csv')
    users = pd.read_csv('Data/Users.csv')
    ratings = pd.read_csv('Data/Ratings.csv')
except FileNotFoundError as e:
    st.error(f"Missing CSV file: {e.filename}")
    books_csv = ratings = users = None

# Sidebar section for showing the dataset
st.sidebar.title("Data Used")

if st.sidebar.button("Show"):
    if books_csv is not None:
        st.subheader('This is the books data we used in our model')
        st.dataframe(books_csv)
    if ratings is not None:
        st.subheader('This is the User ratings data we used in our model')
        st.dataframe(ratings)
    if users is not None:
        st.subheader('This is the user data we used in our model')
        st.dataframe(users)
