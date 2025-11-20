from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = FastAPI()

# Global variables to hold data in memory
data_store = {}

@app.on_event("startup")
def load_data():
    print("Loading datasets... this may take a moment.")
    
    # 1. Load Recommendation Data
    # Assuming columns: 'imdb_id', 'cluster_label_new'
    try:
        df_rec = pd.read_csv("For_recommendation.csv")
        # Convert to string to ensure matching works
        df_rec['imdb_id'] = df_rec['imdb_id'].astype(str)
        # Set index for O(1) lookup speed
        data_store['rec_df'] = df_rec.set_index('imdb_id')
        # Also keep a copy for filtering clusters easily
        data_store['rec_df_reset'] = df_rec 
        print("Recommendation data loaded.")
    except Exception as e:
        print(f"Error loading For_recommendation.csv: {e}")

    # 2. Load Vectors Data
    # Assuming columns: 'imdb_id', 'rounded_vectors'
    try:
        df_vec = pd.read_csv("text_vectors.csv")
        df_vec['imdb_id'] = df_vec['imdb_id'].astype(str)
        
        # Optimization: Convert the string representation of list "[0.1, 0.2]" 
        # into actual python lists immediately.
        # Warning: If dataset is huge, this is slow. 
        print("Parsing vectors...")
        df_vec['rounded_vectors'] = df_vec['rounded_vectors'].apply(
            lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
        )
        
        data_store['vec_df'] = df_vec.set_index('imdb_id')
        print("Vectors data loaded.")
    except Exception as e:
        print(f"Error loading text_vectors.csv: {e}")


@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running."}


@app.get("/recommend")
def recommend(imdb_id: str):
    """
    Endpoint 1: Receives imdb_id, finds its cluster, returns all movies in that cluster.
    """
    df = data_store.get('rec_df')
    df_full = data_store.get('rec_df_reset')
    
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if imdb_id not in df.index:
        raise HTTPException(status_code=404, detail="IMDB ID not found in recommendation database")

    # 1. Get the cluster label for the requested ID
    try:
        target_cluster = df.loc[imdb_id, 'cluster_label_new']
    except KeyError:
        raise HTTPException(status_code=404, detail="ID found but cluster missing")

    # 2. Filter all movies with that cluster
    # (Using the reset index version for boolean indexing)
    similar_movies = df_full[df_full['cluster_label_new'] == target_cluster]
    
    # Return list of IDs (excluding the input ID if you prefer, currently keeping it)
    results = similar_movies['imdb_id'].tolist()
    
    return {
        "input_id": imdb_id,
        "cluster": int(target_cluster),
        "count": len(results),
        "recommendations": results
    }


@app.get("/check_similarity")
def check_similarity(id1: str, id2: str):
    """
    Endpoint 2: Receives 2 IDs, fetches vectors, calculates cosine similarity.
    """
    df_vec = data_store.get('vec_df')
    
    if df_vec is None:
        raise HTTPException(status_code=500, detail="Vector data not loaded")

    # Check existence
    if id1 not in df_vec.index or id2 not in df_vec.index:
        raise HTTPException(status_code=404, detail="One or both IMDB IDs not found in vector database")

    # 1. Fetch Vectors
    vec1 = df_vec.loc[id1, 'rounded_vectors']
    vec2 = df_vec.loc[id2, 'rounded_vectors']

    # 2. Calculate Cosine Similarity
    # Reshape because sklearn expects 2D arrays
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    similarity_score = cosine_similarity(vec1, vec2)[0][0]

    return {
        "id1": id1,
        "id2": id2,
        "cosine_similarity": float(similarity_score)
    }