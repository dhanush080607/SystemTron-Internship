import pandas as pd
import numpy as np
import ast

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

movies = pd.read_csv(r"C:\Users\hdhan\OneDrive\Documents\Python for data science\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\hdhan\OneDrive\Documents\Python for data science\tmdb_5000_credits.csv")

df = movies.merge(credits, on='title')

df = df[['genres', 'keywords', 'cast', 'crew',
         'popularity', 'budget', 'runtime',
         'vote_average']]

df.dropna(inplace=True)

def extract_names(text):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(text)])
    except:
        return ""

def extract_director(text):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director'])
    except:
        return ""

df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['cast'] = df['cast'].apply(extract_names)
df['director'] = df['crew'].apply(extract_director)

df.drop('crew', axis=1, inplace=True)

X = df.drop('vote_average', axis=1)
y = df['vote_average']

preprocessor = ColumnTransformer(
    transformers=[
        ('genres', TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1,2)), 'genres'),
        ('keywords', TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1,2)), 'keywords'),
        ('cast', TfidfVectorizer(max_features=300), 'cast'),
        ('director', TfidfVectorizer(max_features=100), 'director')
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = Ridge(alpha=1.0)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", round(mse, 3))
print("R2 Score:", round(r2, 3))

sample = pd.DataFrame({
    'genres': ['Action Adventure'],
    'keywords': ['hero future world'],
    'cast': ['Tom Cruise Emily Blunt'],
    'director': ['Christopher McQuarrie'],
    'popularity': [120.5],
    'budget': [150000000],
    'runtime': [130]
})

prediction = pipeline.predict(sample)

print("Predicted Rating:", round(prediction[0], 2))