import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz, process
import re
import string


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


# def match_candidates(linkedin_df, naukri_df, selected_health):
#     """
#     Matches candidates from LinkedIn and Naukri data based on multiple criteria.
#     Parameters:
#         linkedin_df (DataFrame): LinkedIn profiles
#         naukri_df (DataFrame): Naukri profiles
#         selected_health (str): Profile health filter (High, Medium, Low)
#     Returns:
#         DataFrame: Matched candidates
#     """
#     linkedin_filtered = linkedin_df[linkedin_df["Profile Health"] == selected_health]
#     linkedin_filtered = linkedin_filtered.copy()

#     # Process experiences and educations dynamically
#     linkedin_filtered["Experiences"] = linkedin_filtered.apply(extract_experiences, axis=1)
#     linkedin_filtered["Educations"] = linkedin_filtered.apply(extract_educations, axis=1)
#     linkedin_filtered["Processed Name"] = linkedin_filtered["Candidate Name"].apply(preprocess_name)

#     naukri_df["Processed Name"] = naukri_df["Candidate Name"].apply(preprocess_name)

#     linkedin_names = linkedin_filtered["Processed Name"].tolist()

#     matches = []
#     for _, naukri_row in naukri_df.iterrows():
#         naukri_name = naukri_row["Processed Name"]
#         if not naukri_name:
#             continue  # Skip if no name

#         # Fuzzy match
#         match, score = process.extractOne(naukri_name, linkedin_names, scorer=fuzz.token_sort_ratio)
#         if match and score >= 85:
#             matched_row = linkedin_filtered[linkedin_filtered["Processed Name"] == match].iloc[0]
#             combined_profile = {**matched_row.to_dict(), **naukri_row.to_dict()}
#             matches.append(combined_profile)

#     return pd.DataFrame(matches)

def match_candidates(linkedin_df, naukri_df, selected_health):
    """
    Matches candidates from LinkedIn and Naukri data based on multiple criteria.
    Parameters:
        linkedin_df (DataFrame): LinkedIn profiles
        naukri_df (DataFrame): Naukri profiles
        selected_health (str): Profile health filter (High, Medium, Low)
    Returns:
        DataFrame: Matched candidates
    """
    linkedin_filtered = linkedin_df[linkedin_df["Profile Health"] == selected_health]

    # If no LinkedIn profiles match the selected health, return an empty DataFrame
    if linkedin_filtered.empty:
        return pd.DataFrame()

    linkedin_filtered = linkedin_filtered.copy()

    # Process experiences and educations dynamically
    linkedin_filtered["Experiences"] = linkedin_filtered.apply(extract_experiences, axis=1)
    linkedin_filtered["Educations"] = linkedin_filtered.apply(extract_educations, axis=1)
    linkedin_filtered["Processed Name"] = linkedin_filtered["Candidate Name"].apply(preprocess_name)

    naukri_df["Processed Name"] = naukri_df["Candidate Name"].apply(preprocess_name)

    linkedin_names = linkedin_filtered["Processed Name"].tolist()

    # If linkedin_names is empty, return an empty DataFrame
    if not linkedin_names:
        return pd.DataFrame()

    matches = []
    for _, naukri_row in naukri_df.iterrows():
        naukri_name = naukri_row["Processed Name"]
        if not naukri_name:
            continue  # Skip if no name

        # Fuzzy match
        match_result = process.extractOne(naukri_name, linkedin_names, scorer=fuzz.token_sort_ratio)
        if match_result is not None:
            match, score = match_result
            if score >= 85:
                matched_row = linkedin_filtered[linkedin_filtered["Processed Name"] == match].iloc[0]
                combined_profile = {**matched_row.to_dict(), **naukri_row.to_dict()}
                matches.append(combined_profile)

    return pd.DataFrame(matches)

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
