from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

app = FastAPI()

# Global variables to hold data in memory
data_store = {}

def parse_vector(s):
    """
    Parses a numpy-style string representation of an array.
    Input: "[ -1.2   0.63  -0.04 ... ]"
    Output: np.array([-1.2, 0.63, -0.04, ...])
    """
    if not isinstance(s, str):
        return s
    
    # 1. Remove brackets
    s = s.strip('[]')
    
    # 2. Replace newlines with spaces (just in case)
    s = s.replace('\n', ' ')
    
    # 3. Split by whitespace (automatically handles multiple spaces)
    # 4. Convert to float
    try:
        return np.array([float(x) for x in s.split() if x])
    except ValueError:
        return np.zeros(1) # Fallback for bad data

@app.on_event("startup")
def load_data():
    print("Loading datasets...")
    
    # 1. Load Recommendation Data
    try:
        df_rec = pd.read_csv("For_recommendation.csv")
        df_rec['imdb_id'] = df_rec['imdb_id'].astype(str)
        data_store['rec_df'] = df_rec.set_index('imdb_id')
        data_store['rec_df_reset'] = df_rec 
        print("Recommendation data loaded.")
    except Exception as e:
        print(f"Error loading For_recommendation.csv: {e}")

    # 2. Load Vectors Data
    try:
        df_vec = pd.read_csv("text_vectors.csv")
        df_vec['imdb_id'] = df_vec['imdb_id'].astype(str)
        
        print("Parsing vectors (Space-separated format)...")
        # Apply the new parsing function
        df_vec['rounded_vectors'] = df_vec['rounded_vectors'].apply(parse_vector)
        
        data_store['vec_df'] = df_vec.set_index('imdb_id')
        print("Vectors data loaded.")
    except Exception as e:
        print(f"Error loading text_vectors.csv: {e}")


@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running."}


@app.get("/recommend")
def recommend(imdb_id: str):
    df = data_store.get('rec_df')
    df_full = data_store.get('rec_df_reset')
    
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if imdb_id not in df.index:
        raise HTTPException(status_code=404, detail="IMDB ID not found in recommendation database")

    # 1. Get the cluster
    try:
        target_cluster = df.loc[imdb_id, 'cluster_label_new']
    except KeyError:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # 2. Filter movies in the same cluster
    cluster_movies = df_full[df_full['cluster_label_new'] == target_cluster]

    # 3. Remove the input movie itself
    cluster_movies = cluster_movies[cluster_movies['imdb_id'] != imdb_id]

    # 4. Select 8 movies
    if len(cluster_movies) > 8:
        recommendations = cluster_movies.sample(n=8)
    else:
        recommendations = cluster_movies

    results = recommendations['imdb_id'].tolist()
    
    return {
        "input_id": imdb_id,
        "cluster": int(target_cluster),
        "count": len(results),
        "recommendations": results
    }


@app.get("/check_similarity")
def check_similarity(id1: str, id2: str):
    df_vec = data_store.get('vec_df')
    
    if df_vec is None:
        raise HTTPException(status_code=500, detail="Vector data not loaded")

    if id1 not in df_vec.index or id2 not in df_vec.index:
        raise HTTPException(status_code=404, detail="IMDB ID not found")

    # Fetch vectors
    vec1 = df_vec.loc[id1, 'rounded_vectors']
    vec2 = df_vec.loc[id2, 'rounded_vectors']

    # Ensure they are valid arrays before reshaping
    if isinstance(vec1, (int, float)) or isinstance(vec2, (int, float)):
         raise HTTPException(status_code=500, detail="Error parsing vector data for these IDs")

    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    similarity_score = cosine_similarity(vec1, vec2)[0][0]

    return {
        "id1": id1,
        "id2": id2,
        "cosine_similarity": float(similarity_score)
    }