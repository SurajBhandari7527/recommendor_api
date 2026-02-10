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


## 🛣️ How to use (Endpoints)

### 1. Get Recommendations
Find movies similar to a specific IMDb ID.
*   **URL:** `https://recommendor-api-2.onrender.com/recommend?imdb_id=YOUR_ID`
*   **Example:** `https://recommendor-api-2.onrender.com/recommend?imdb_id=tt5950044`

### 2. Check Similarity
Compare two movies to see a similarity score (between 0 and 1).
*   **URL:** `https://recommendor-api-2.onrender.com/check_similarity?id1=ID_1&id2=ID_2`
*   **Example:** `https://recommendor-api-2.onrender.com/check_similarity?id1=tt5950044&id2=tt0081573`

### 3. Home
Check if the API is running.
*   **URL:** `/`

## ⚙️ How it works
*   **Startup:** When the app starts, it loads your CSV files into memory for fast access.
*   **Clustering:** It groups movies that are similar. When you ask for a recommendation, it picks random movies from the same group.
*   **Async Fetching:** To stay fast, it fetches movie details (like posters) for all 8 recommendations at the same time using `asyncio`.
