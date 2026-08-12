# 🎬 Movie Recommendation API

This is a **FastAPI** application that suggests movies based on machine learning clusters and calculates similarity between movies using pre-computed text vectors.

## 🚀 What it does
1.  **Recommends Movies:** Give it a Movie ID (IMDb ID), and it finds 8 similar movies from the same "cluster."
2.  **Enriches Data:** It automatically talks to an external "ID-to-Title" service to get titles, posters, and cast info for the recommendations.
3.  **Similarity Check:** It can compare two movies and tell you exactly how mathematically similar they are (using Cosine Similarity).

## 🛠️ Requirements
Make sure you have the following files in your project folder:
*   `For_recommendation.csv`: Contains movie IDs and their cluster labels.
*   `text_vectors.csv`: Contains the mathematical vectors for each movie.

## How was it developed (Data science Part)
Dataset used: https://www.kaggle.com/code/chanchal24/tmdb-movies-dataset

#### Data Assessing and EDA: Understood the columns and missing data. Since, we aren't building predictive model, so we don't need to understand the relationship and effects between the features that much.

### Feature Engineering: Feature selection was the most important aspect of this project to get the vectors of each movie, that contains overall context of the movie. Using the domain knowledge, I included following features:

| Feature | Reason |
| :--- | :--- |
| Movie Name | If a person likes Spiderman, it is more likely that they will like Spider man 2 |
| Production Companies | If a person likes a production house, he/she might like all of its movies. |
| Rating | Some people only like highest rated movies. |
| Director | If a person likes specific director's movies, he/she might like other of his/her movies. |
| Original Language | If a person loves English movies, he should be recommended movies in English language only. |
| Genres | This is the most important feature, most people love only specific genres movies. |
| tagline | This is one line set up of the movie that captures a lot of information about movie. |
| overview | This is also very important feature for collaborative recommendation. |

### Preprocessing:
1. Extract only 1 production company, if null then use missing indicator imputation 'None'
3. Extract only 1 director name, if more than two. If null then use missing indicator imputation 'None'
4. Extract all the Genres, if null impute 'None'
5. Combine tagline + overview for better context of movie
6. Finally combine all extracted features into a single column

### NLP Preprocesssing: 
1. Convert all the text into lower form.
2. Remove all the stop words.
3. Tokenize and Lemmatize the text.
4. Vectorize it using word2vec to get vectors for each movie.
   
   Output: text_vectors.csv

Notebook file: Notebooks/vectorization_of_movies.ipynb

## Clustering technique:

### Problem:
Now that, we have context vectors for each movie we can recommend the movies based on a single movie but calculating cosine similarity score with 5 lakh+ movies and sorting them to get top 8 is computationally very high.

### Solution:
I used clustering, to assign the movies to different clusters where each cluster's movies will be similar. So, we don't need to calculate similarity between 5 lakh different movies for a recommendation, but we can just filter out the movies of cluster and calculate similarity between them and sort in descending to recommend the best from the cluster.

Output: For_recommendation.csv
  
Notebook file: Notebooks/clustering_the_movies.ipynb

## 🛣️ How to use (Endpoints)

### 1. Get Recommendations
Find movies similar to a specific IMDb ID.
*   **URL:** `https://recommendor-api-2.onrender.com/recommend?imdb_id=YOUR_ID`
*   **Example:** `https://recommendor-api-2.onrender.com/recommend?imdb_id=tt5950044`

### 3. Home
Check if the API is running.
*   **URL:** https://recommendor-api-2.onrender.com/recommend?imdb_id=tt5950044


## ⚙️ How it works
*   **Startup:** When the app starts, it loads your CSV files into memory for fast access.
*   **Clustering:** It groups movies that are similar. When you ask for a recommendation, it picks random movies from the same group.
*   **Async Fetching:** To stay fast, it fetches movie details (like posters) for all 8 recommendations at the same time using `asyncio`.
