from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import httpx
import asyncio

app = FastAPI()

# Global variables to hold data in memory
data_store = {}

def parse_vector(s):
    """
    Parses a numpy-style string representation of an array.
    Input: "[ -1.2   0.63  -0.04 ... ]"
    Output: np.array([-1.2, 0.63, ...])
    """
    if not isinstance(s, str):
        return s
    s = s.strip('[]').replace('\n', ' ')
    try:
        return np.array([float(x) for x in s.split() if x])
    except ValueError:
        return np.zeros(1) 

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
        print("Parsing vectors...")
        df_vec['rounded_vectors'] = df_vec['rounded_vectors'].apply(parse_vector)
        data_store['vec_df'] = df_vec.set_index('imdb_id')
        print("Vectors data loaded.")
    except Exception as e:
        print(f"Error loading text_vectors.csv: {e}")

    # 3. NEW: Load Title Mapping
    try:
        df_titles = pd.read_csv("imdb_id_to_title.csv")
        # Ensure columns are correct strings
        df_titles['imdb_id'] = df_titles['imdb_id'].astype(str)
        # Create a dictionary for O(1) lookup: {'tt123': 'Batman', ...}
        data_store['title_map'] = dict(zip(df_titles['imdb_id'], df_titles['title']))
        print("Title mapping loaded.")
    except Exception as e:
        print(f"Error loading imdb_id_to_title.csv: {e}")


@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running."}


@app.get("/get_title")
def get_title(imdb_id: str):
    """
    Simple endpoint to convert ID -> Title
    """
    title_map = data_store.get('title_map', {})
    title = title_map.get(imdb_id)
    
    if not title:
        raise HTTPException(status_code=404, detail="Title not found for this ID")
        
    return {"imdb_id": imdb_id, "title": title}


async def fetch_movie_poster(client, imdb_id, title):
    """
    Helper function to call external API and filter the result.
    """
    if not title:
        return None

    external_url = "https://id-to-poster-api-3.onrender.com/search_movie"
    
    try:
        # Call external API with the Title
        response = await client.get(external_url, params={"query": title})
        
        if response.status_code == 200:
            search_results = response.json()
            
            # The API returns a list of movies (Batman, Batman Returns, etc.)
            # We must find the one that matches our specific imdb_id
            for movie in search_results:
                if movie.get('imdb_id') == imdb_id:
                    return movie # Found the exact match with poster
            
            # Fallback: If exact ID match fails in external results, 
            # return the first result or just local data
            return {"imdb_id": imdb_id, "title": title, "poster_path": None, "cast": "Not found in external API"}
            
    except Exception as e:
        print(f"External API Error for {title}: {e}")
        return None
    
    return {"imdb_id": imdb_id, "title": title, "poster_path": None}


@app.get("/recommend")
async def recommend(imdb_id: str):
    # Note: Changed to 'async def' to allow parallel API calls
    
    df = data_store.get('rec_df')
    df_full = data_store.get('rec_df_reset')
    title_map = data_store.get('title_map', {})
    
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if imdb_id not in df.index:
        raise HTTPException(status_code=404, detail="IMDB ID not found in recommendation database")

    # 1. Get the cluster
    try:
        target_cluster = df.loc[imdb_id, 'cluster_label_new']
    except KeyError:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # 2. Filter movies in the same cluster & remove input ID
    cluster_movies = df_full[df_full['cluster_label_new'] == target_cluster]
    cluster_movies = cluster_movies[cluster_movies['imdb_id'] != imdb_id]

    # 3. Select 8 movies
    if len(cluster_movies) > 8:
        recommendations_df = cluster_movies.sample(n=8)
    else:
        recommendations_df = cluster_movies

    recommended_ids = recommendations_df['imdb_id'].tolist()

    # 4. Fetch Posters/Details for all 8 movies in parallel
    # We use httpx.AsyncClient to make simultaneous requests
    enriched_results = []
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for rec_id in recommended_ids:
            rec_title = title_map.get(rec_id)
            # Schedule the task
            tasks.append(fetch_movie_poster(client, rec_id, rec_title))
        
        # Run all 8 requests at the same time
        results = await asyncio.gather(*tasks)
        
        # Filter out any None results (failed requests)
        enriched_results = [r for r in results if r is not None]

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
        raise HTTPException(status_code=404, detail="IMDB ID not found")

    vec1 = df_vec.loc[id1, 'rounded_vectors']
    vec2 = df_vec.loc[id2, 'rounded_vectors']

    if isinstance(vec1, (int, float)) or isinstance(vec2, (int, float)):
         raise HTTPException(status_code=500, detail="Error parsing vector data")

    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    similarity_score = cosine_similarity(vec1, vec2)[0][0]

    return {
        "id1": id1,
        "id2": id2,
        "cosine_similarity": float(similarity_score)
    }