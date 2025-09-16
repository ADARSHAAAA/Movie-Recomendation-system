

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

movies = pd.read_csv('/content/tmdb_5000_credits.csv')
credits = pd.read_csv('/content/tmdb_5000_movies.csv') 

movies = movies.merge(credits,on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

import ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies.head()

movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

movies['cast'] = movies['cast'].apply(convert)
movies.head()

movies['cast'] = movies['cast'].apply(lambda x:x[0:3])

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

movies['crew'] = movies['crew'].apply(fetch_director)

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

new[new['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

import pickle

pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


import pickle
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import logging

logging.basicConfig(level=logging.WARNING)  # thread logs go here

PLACEHOLDER = "https://via.placeholder.com/500x750?text=No+Image"

def create_session_with_retries(total_retries=4, backoff_factor=0.6):
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# NOTE: This function MUST NOT call any Streamlit APIs (st.*)
def fetch_poster_no_st(movie_id: int) -> tuple:
    """
    Return (poster_url, error_message_or_None).
    No Streamlit calls inside this function.
    """
    session = create_session_with_retries()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return ("https://image.tmdb.org/t/p/w500/" + poster_path, None)
        else:
            return (PLACEHOLDER, None)
    except Exception as e:
        # Log from background thread (safe)
        logging.warning(f"fetch_poster failed for ID {movie_id}: {e}")
        return (PLACEHOLDER, str(e))


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    top = distances[1:6]  # 5 recommendations
    movie_ids = [movies.iloc[i[0]].movie_id for i in top]
    movie_titles = [movies.iloc[i[0]].title for i in top]

    # fetch posters in parallel (no st.* calls in workers)
    posters = []
    errors = []  # collect any errors to show in main thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_poster_no_st, mid): mid for mid in movie_ids}
        for fut in concurrent.futures.as_completed(futures):
            mid = futures[fut]
            poster_url, error = fut.result()
            posters.append((movie_ids.index(mid), poster_url))  # keep position
            if error:
                errors.append((mid, error))

    # reorder posters by original order
    posters_sorted = [p for _, p in sorted(posters, key=lambda x: x[0])]
    return movie_titles, posters_sorted, errors


# --- Streamlit UI ---
st.header('Movie Recommender System')

movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    names, posters, errors = recommend(selected_movie)

    # If there were background errors, show one aggregated warning (main thread)
    if errors:
        # Build a short message (avoid flooding UI)
        err_msg = "Some posters failed to fetch for movie IDs: " + ", ".join(str(e[0]) for e in errors)
        st.warning(err_msg)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            st.image(poster)
