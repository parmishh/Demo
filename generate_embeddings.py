# generate_embeddings.py

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("courses.csv")

def get_course_text(row):
    parts = [str(row.get(col, '')) for col in ['course_title', 'organization', 'skills', 'description']]
    return " ".join(parts)

df['combined_text'] = df.apply(get_course_text, axis=1)
embeddings = model.encode(df['combined_text'].tolist())

with open("embeddings.pickle", "wb") as f:
    pickle.dump((df, embeddings), f)

print("✅ Embeddings saved to embeddings.pickle")
