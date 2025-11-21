from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import httpx
import asyncio

app = FastAPI()

# --- CONFIGURATION ---
# URL of your independent ID-to-Details service
TITLE_API_URL = "https://id-to-title.onrender.com/get_title"

# Global Data Store
data_store = {}

def parse_vector(s):
    """
    Parses specific numpy-style string format: "[ -1.2   0.5 ... ]"
    """
    if not isinstance(s, str):
        return s
    # Remove brackets and replace newlines with spaces
    s = s.strip('[]').replace('\n', ' ')
    try:
        # Split by whitespace
        return np.array([float(x) for x in s.split() if x], dtype=np.float32)
    except ValueError:
        return np.zeros(1, dtype=np.float32)

@app.on_event("startup")
def load_data():
    print("Loading datasets...")
    
    # 1. Load Recommendation Clusters
    try:
        df_rec = pd.read_csv("For_recommendation.csv", dtype={'imdb_id': str})
        if 'cluster_label_new' in df_rec.columns:
             df_rec['cluster_label_new'] = pd.to_numeric(df_rec['cluster_label_new'], errors='coerce')
        
        data_store['rec_df'] = df_rec.set_index('imdb_id')
        data_store['rec_df_reset'] = df_rec 
        print("Recommendation data loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR loading For_recommendation.csv: {e}")

    # 2. Load Vectors
    try:
        df_vec = pd.read_csv("text_vectors.csv", usecols=['imdb_id', 'rounded_vectors'], dtype={'imdb_id': str})
        print("Parsing vectors...")
        df_vec['rounded_vectors'] = df_vec['rounded_vectors'].apply(parse_vector)
        
        data_store['vec_df'] = df_vec.set_index('imdb_id')
        print("Vectors data loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR loading text_vectors.csv: {e}")


async def fetch_movie_details(client, imdb_id):
    """
    Simply calls the Title API with the ID and returns whatever JSON it gives.
    It returns {imdb_id, title, cast, poster_path} directly.
    """
    try:
        # Call the external API
        response = await client.get(TITLE_API_URL, params={"imdb_id": imdb_id})
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
             return {"imdb_id": imdb_id, "error": "Details not found in database"}
        else:
            return {"imdb_id": imdb_id, "error": f"API Error {response.status_code}"}

    except httpx.TimeoutException:
        return {"imdb_id": imdb_id, "error": "Request Timed Out"}
    except Exception as e:
        return {"imdb_id": imdb_id, "error": str(e)}


@app.get("/")
def home():
    return {"message": "Main Recommendation API is running."}


@app.get("/recommend")
async def recommend(imdb_id: str):
    df = data_store.get('rec_df')
    df_full = data_store.get('rec_df_reset')
    
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if imdb_id not in df.index:
        raise HTTPException(status_code=404, detail="IMDB ID not found in database")

    # 1. Find Cluster
    try:
        target_cluster = df.loc[imdb_id, 'cluster_label_new']
    except KeyError:
        raise HTTPException(status_code=404, detail="Cluster label missing")

    # 2. Get candidate IDs from that cluster
    cluster_movies = df_full[df_full['cluster_label_new'] == target_cluster]
    
    # Remove the input movie itself from recommendations
    cluster_movies = cluster_movies[cluster_movies['imdb_id'] != imdb_id]

    # 3. Sample 8 movies
    if len(cluster_movies) > 8:
        recommendations_df = cluster_movies.sample(n=8)
    else:
        recommendations_df = cluster_movies

    rec_ids = recommendations_df['imdb_id'].tolist()

    # 4. Async Parallel Fetch
    # We call your other API for all 8 IDs at the same time
    enriched_results = []
    
    # Timeout set to 30s to handle Render cold starts
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [fetch_movie_details(client, rid) for rid in rec_ids]
        enriched_results = await asyncio.gather(*tasks)

    return {
        "input_id": imdb_id,
        "cluster": int(target_cluster),
        "count": len(enriched_results),
        "recommendations": enriched_results
    }


@app.get("/check_similarity")
def check_similarity(id1: str, id2: str):
    df_vec = data_store.get('vec_df')
    
    if df_vec is None:
        raise HTTPException(status_code=500, detail="Vector data not loaded")

    if id1 not in df_vec.index or id2 not in df_vec.index:
        raise HTTPException(status_code=404, detail="One or both IMDB IDs not found")

    # Extract vectors
    v1 = df_vec.loc[id1, 'rounded_vectors']
    v2 = df_vec.loc[id2, 'rounded_vectors']

    # Safety check
    if isinstance(v1, (int, float)) or isinstance(v2, (int, float)):
         raise HTTPException(status_code=500, detail="Vectors corrupted or missing")

    # Reshape for sklearn
    v1 = v1.reshape(1, -1)
    v2 = v2.reshape(1, -1)
    
    score = cosine_similarity(v1, v2)[0][0]

    return {
        "id1": id1,
        "id2": id2,
        "cosine_similarity": float(score)
    }