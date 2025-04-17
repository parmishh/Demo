import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Load raw course data
df = pd.read_csv("data/courses.csv")

# Check the columns to verify
print("Columns in CSV:", df.columns)

# Clean skills column (if needed)
df["course_skills"] = df["course_skills"].apply(
    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else []
)

# Combine all searchable text into one column
df["search_blob"] = (
    df["course_title"].fillna("") + " " +
    df["course_description"].fillna("") + " " +  # Make sure this column exists in your CSV
    df["course_organization"].fillna("") + " " +
    df["course_time"].fillna("") + " " +
    df["course_skills"].apply(lambda skills: " ".join(skills))  # Join skills into a string
)

# Load embedding model (small & fast)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all course blobs
embeddings = model.encode(df["search_blob"].tolist(), show_progress_bar=True)

# Save the embeddings and the data
with open("data/embedded.pkl", "wb") as f:
    pickle.dump((df, embeddings), f)

print("✅ Data and embeddings saved to data/embedded.pkl")
