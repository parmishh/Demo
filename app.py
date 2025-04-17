import streamlit as st
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Streamlit page config
# st.set_page_config(page_title="Learning Path Recommender", layout="centered")

# def set_background(image_path):
#     with open(image_path, "rb") as img_file:
#         img_base64 = base64.b64encode(img_file.read()).decode()
#     background_style = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{img_base64}");
#         background-size: cover;
#         background-position: center;
#     }}
#     </style>
#     """
#     st.markdown(background_style, unsafe_allow_html=True)

# # Call the function in your script
# set_background("img1.jpg")

# Load model and embeddings from cache
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_embeddings():
    with open("embeddings.pickle", "rb") as f:
        df, embeddings = pickle.load(f)
    return df, embeddings

# Function to get recommendations based on cosine similarity
def get_recommendations(user_input, df, embeddings, model, top_k=7):
    user_vector = model.encode([user_input])  # Get the user query embedding
    similarities = cosine_similarity(user_vector, embeddings)[0]  # Calculate cosine similarities
    df['similarity'] = similarities  # Add similarity to DataFrame
    results = df.sort_values(by="similarity", ascending=False).head(top_k)  # Get top_k recommendations
    return results

# Function to compute difficulty score based on course duration and required skills
def compute_difficulty(row):
    # Map course duration to difficulty score (longer = harder)
    duration = row['course_time']
    if "3 - 6 Months" in duration:
        difficulty = 2
    elif "6 - 12 Months" in duration:
        difficulty = 3
    else:
        difficulty = 1  # Default to easy if duration is less than 3 months

    # Add difficulty based on skills required (if advanced skills are mentioned)
    advanced_skills = ["machine learning", "deep learning", "data science", "AI", "statistical analysis"]
    if any(skill in row['course_skills'].lower() for skill in advanced_skills):
        difficulty += 1  # Increase difficulty for advanced skills
    
    return difficulty

# Load model and embeddings during page load (not after search)
model = load_model()
df, embeddings = load_embeddings()

# Streamlit UI for user input
st.title("🧭 Personalized Learning Path")
designation = st.text_input("Your Designation", "")
department = st.text_input("Your Department", "")

# Show recommendations if both fields are filled
if designation and department:
    st.markdown("### 🎯 Recommended Courses Learning Path")
    user_query = f"{designation} {department}"  # Combine user inputs into a query

    # Get the recommendations based on user input
    results = get_recommendations(user_query, df, embeddings, model)

    # Compute difficulty for each recommended course
    results['difficulty'] = results.apply(compute_difficulty, axis=1)

    # Sort results by difficulty in ascending order (easier courses first)
    sorted_results = results.sort_values(by='difficulty', ascending=True)

    # Display the sorted recommendations with indexing
    for idx, (_, row) in enumerate(sorted_results.iterrows(), start=1):
        st.subheader(f"Course_{idx}: {row['course_title']}")
        st.markdown(f"🏫 **Organization:** {row['course_organization']}")
        st.markdown(f"⏱️ **Duration:** {row['course_time']}")
        if pd.notna(row.get('course_skills', '')):
            st.markdown(f"🧠 **Skills:** {row['course_skills']}")
        st.markdown(f"🔍 **Similarity Score:** `{row['similarity']:.4f}`")
        st.markdown(f"👉 [Go to Course]({row['course_url']})")
        st.markdown(f"📝 **Difficulty Score:** {row['difficulty']} (Lower is easier)")
        st.markdown("---")


