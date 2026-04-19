import pandas as pd
import numpy as np
import json
import random
import os

def generate_mock_movie_data(num_samples=3000):
    print("Generating synthetic movie dataset...")
    np.random.seed(42)
    random.seed(42)
    
    genres_list = ["Action", "Adventure", "Comedy", "Drama", "Science Fiction", "Horror", "Romance", "Thriller", "Fantasy", "Animation"]
    actors_list = [f"Actor {i}" for i in range(1, 150)]
    
    data = []
    for i in range(num_samples):
        # Budget ranges from 1M to 200M
        budget = max(1000000, np.random.normal(40000000, 35000000))
        
        # Random genres
        g_count = random.randint(1, 4)
        movie_genres = random.sample(genres_list, g_count)
        genres_json = json.dumps([{"id": random.randint(10, 100), "name": g} for g in movie_genres])
        
        # Random actors
        a_count = random.randint(3, 10)
        movie_actors = random.sample(actors_list, a_count)
        cast_json = json.dumps([{"name": a, "character": f"Character {random.randint(1,100)}"} for a in movie_actors])
        
        # Simulated Revenue generation (budget + some random noise + actor bonuses + genre bonuses)
        # Action/Sci-Fi tends to gross more.
        genre_multiplier = 1.0 + (0.5 if "Action" in movie_genres else 0) + (0.4 if "Science Fiction" in movie_genres else 0)
        
        # Popular actors bonus
        actor_bonus = sum([0.1 for a in movie_actors if int(a.split(" ")[1]) < 20])
        
        # Base revenue multiplier
        multiplier = np.random.uniform(0.2, 4.0) + genre_multiplier + actor_bonus
        
        # Final Revenue
        revenue = budget * multiplier
        
        # Introduce some outliers (flops and blockbusters)
        if random.random() < 0.05:
            revenue *= 2.5 # Blockbuster
        if random.random() < 0.05:
            revenue *= 0.1 # Flop
            
        data.append({
            "id": int(i),
            "title": f"Mock Movie {i}",
            "budget": float(budget),
            "genres": genres_json,
            "cast": cast_json,
            "revenue": float(revenue)
        })
        
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/movie_dataset.csv", index=False)
    print("Dataset generated successfully at data/movie_dataset.csv")

if __name__ == "__main__":
    generate_mock_movie_data()
