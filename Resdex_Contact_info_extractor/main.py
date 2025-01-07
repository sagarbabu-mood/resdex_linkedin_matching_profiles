import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz, process
import re
import string
from sentence_transformers import SentenceTransformer
import numpy as np

def preprocess_name(name):
    """
    Preprocesses the candidate name by removing punctuation, converting to lowercase,
    and stripping leading/trailing spaces.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    translator = str.maketrans("", "", string.punctuation)
    return name.translate(translator).lower().strip()

def extract_experiences(linkedin_row):
    """
    Dynamically extracts all experience details from the LinkedIn row.
    Assumes 'Exp1', 'Exp2', ..., 'ExpN' represent experiences in reverse chronological order.
    Returns a list of dictionaries, each containing:
    - Current Employer
    - Designation
    - Experience (in years)
    - Location
    """
    experiences = []
    exp_columns = sorted([col for col in linkedin_row.index if col.startswith("Exp")])

    for exp_col in exp_columns:
        exp_details = linkedin_row.get(exp_col, "")
        if pd.isna(exp_details) or not isinstance(exp_details, str) or not exp_details.strip():
            continue  # Skip empty experience entries

        exp_info = {
            "Current Employer": "",
            "Designation": "",
            "Experience": "",
            "Location": "",
        }
        # Extract relevant details using regex or string parsing
        lines = exp_details.split("\n")
        for i, line in enumerate(lines):
            if "Company name" in line:
                exp_info["Current Employer"] = lines[i + 1].strip() if i + 1 < len(lines) else ""
            elif "Position title" in line:
                exp_info["Designation"] = lines[i + 1].strip() if i + 1 < len(lines) else ""
            elif "Dates employed and Duration" in line:
                exp_info["Experience"] = lines[i + 1].strip() if i + 1 < len(lines) else ""
            elif "Position location" in line:
                exp_info["Location"] = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if exp_info["Current Employer"] or exp_info["Designation"]:
            experiences.append(exp_info)

    return experiences

def extract_educations(linkedin_row):
    """
    Dynamically extracts all education details from the LinkedIn row.
    Assumes 'Edu1', 'Edu2', ..., 'EduN' represent education entries in reverse chronological order.
    Returns a list of dictionaries with education details.
    """
    educations = []
    edu_columns = sorted([col for col in linkedin_row.index if col.startswith("Edu")])

    for edu_col in edu_columns:
        edu_details = linkedin_row.get(edu_col, "")
        if pd.isna(edu_details) or not isinstance(edu_details, str) or not edu_details.strip():
            continue  # Skip empty education entries

        educations.append({"Education Details": edu_details.strip()})

    return educations

def get_profile_embedding(profile_data):
    """
    Creates an embedding for a candidate's profile by combining relevant information.
    """
    # Combine relevant profile information
    profile_text = f"{profile_data['Candidate Name']} "
    
    # Add experience information
    if isinstance(profile_data.get('Experiences'), list):
        for exp in profile_data['Experiences']:
            profile_text += f"{exp.get('Current Employer', '')} {exp.get('Designation', '')} "
    
    # Add education information
    if isinstance(profile_data.get('Educations'), list):
        for edu in profile_data['Educations']:
            profile_text += f"{edu.get('Education Details', '')} "
    
    return profile_text.strip()

def match_candidates(linkedin_df, naukri_df, selected_health, similarity_threshold=0.85):
    """
    Matches candidates using AI embeddings for better accuracy.
    """
    linkedin_filtered = linkedin_df[linkedin_df["Profile Health"] == selected_health].copy()
    
    if linkedin_filtered.empty:
        return pd.DataFrame()

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process experiences and educations
    linkedin_filtered["Experiences"] = linkedin_filtered.apply(extract_experiences, axis=1)
    linkedin_filtered["Educations"] = linkedin_filtered.apply(extract_educations, axis=1)
    
    # Create embeddings for LinkedIn profiles
    linkedin_texts = linkedin_filtered.apply(get_profile_embedding, axis=1).tolist()
    linkedin_embeddings = model.encode(linkedin_texts, convert_to_tensor=True)
    
    matches = []
    
    # Process Naukri profiles in batches
    batch_size = 32
    for i in range(0, len(naukri_df), batch_size):
        batch = naukri_df.iloc[i:i+batch_size]
        
        # Create embeddings for Naukri profiles
        naukri_texts = batch.apply(get_profile_embedding, axis=1).tolist()
        naukri_embeddings = model.encode(naukri_texts, convert_to_tensor=True)
        
        # Calculate similarity scores
        similarity_scores = np.inner(naukri_embeddings, linkedin_embeddings)
        
        # Find matches above threshold
        for idx, scores in enumerate(similarity_scores):
            max_score = np.max(scores)
            if max_score >= similarity_threshold:
                linkedin_idx = np.argmax(scores)
                naukri_row = batch.iloc[idx]
                linkedin_row = linkedin_filtered.iloc[linkedin_idx]
                
                combined_profile = {
                    **linkedin_row.to_dict(),
                    **naukri_row.to_dict(),
                    'Match_Score': float(max_score)
                }
                matches.append(combined_profile)
    
    matched_df = pd.DataFrame(matches)
    if not matched_df.empty:
        matched_df = matched_df.sort_values('Match_Score', ascending=False)
    
    return matched_df

def main():
    st.title("Dynamic Candidate Matching App")
    st.write("Upload LinkedIn and Naukri CSV files to match candidates dynamically.")

    linkedin_file = st.file_uploader("Upload LinkedIn CSV", type=["csv"])
    naukri_file = st.file_uploader("Upload Naukri CSV", type=["csv"])

    if linkedin_file and naukri_file:
        linkedin_df = pd.read_csv(linkedin_file)
        naukri_df = pd.read_csv(naukri_file)

        st.write("Sample Data")
        st.write("LinkedIn Data:")
        st.dataframe(linkedin_df.head())
        st.write("Naukri Data:")
        st.dataframe(naukri_df.head())

        profile_health = st.selectbox("Select Profile Health", ["High", "Medium", "Low"])
        if st.button("Match Candidates"):
            matched_df = match_candidates(linkedin_df, naukri_df, profile_health)
            if(not matched_df.empty):
                st.write("Matched Profiles:")
                st.dataframe(matched_df)
                # Allow download of matched profiles
                st.download_button(
                    "Download Matched Candidates",
                    data=matched_df.to_csv(index=False),
                    file_name="matched_candidates.csv",
                    mime="text/csv",
                )
            
            else:
                st.write("No Matched Profiles")

if __name__ == "__main__":
    main()
