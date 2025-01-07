from fuzzywuzzy import fuzz
from fuzzywuzzy import process
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
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    name_clean = name.translate(translator).lower().strip()
    return name_clean
def fuzzy_match(naukri_name, linkedin_names, threshold=85):
    """
    Matches a Naukri name with a list of LinkedIn names using fuzzy matching.
    Returns the LinkedIn name and the matching score if a match is found, otherwise (None, 0).
    """
    match, score = process.extractOne(naukri_name, linkedin_names, scorer=fuzz.token_sort_ratio)
    return (match, score) if score >= threshold else (None, 0)
    translator = str.maketrans("", "", string.punctuation)
    return name.translate(translator).lower().strip()

def extract_experience_duration(exp_str):
    """
    Extracts numerical experience in years from a string.
    For example:
    - "2 yrs 5 mos" -> 2.42 years
    - "3y 0m" -> 3.0 years
    - "Sep 2022 – Present • 2 yrs 5 mos" -> 2.42 years
    """
    if pd.isna(exp_str) or not isinstance(exp_str, str):
        return 0.0
    years = 0.0
    months = 0.0
    # Extract years and months using regex
    year_match = re.search(r'(\d+)\s*y', exp_str.lower())
    month_match = re.search(r'(\d+)\s*m', exp_str.lower())
    if year_match:
        years = float(year_match.group(1))
    if month_match:
        months = float(month_match.group(1))
    return years + (months / 12)

def extract_experiences(linkedin_row):
    """
    Extracts all experience details from the LinkedIn row.
    Assumes 'Exp1', 'Exp2', ..., 'ExpN' are present and represent experiences in reverse chronological order.
    
    Dynamically extracts all experience details from the LinkedIn row.
    Assumes 'Exp1', 'Exp2', ..., 'ExpN' represent experiences in reverse chronological order.
    Returns a list of dictionaries, each containing:
    - Current Employer
    - Designation
    - Experience (in years)
    - Location
    """
    experiences = []
    # Identify all experience columns (Exp1, Exp2, ...)
    exp_columns = [col for col in linkedin_row.index if col.startswith('Exp')]
    exp_columns = sorted([col for col in linkedin_row.index if col.startswith("Exp")])
    for exp_col in exp_columns:
        exp_details = linkedin_row.get(exp_col, "")
        if pd.isna(exp_details) or not isinstance(exp_details, str) or not exp_details.strip():
            continue  # Skip empty experience entries
        
        # Extract relevant details using regex or string parsing
        lines = exp_details.split('\n')
        exp_info = {
            "Current Employer": "",
            "Designation": "",
            "Experience": 0.0,
            "Location": "",
            "Experience": "",
            "Location": "",
        }
        # Extract relevant details using regex or string parsing
        lines = exp_details.split("\n")
        for i, line in enumerate(lines):
            if "Company name" in line:
                if i + 1 < len(lines):
                    exp_info["Current Employer"] = lines[i + 1].strip()
                exp_info["Current Employer"] = lines[i + 1].strip() if i + 1 < len(lines) else ""
            elif "Position title" in line:
                if i + 1 < len(lines):
                    exp_info["Designation"] = lines[i + 1].strip()
                exp_info["Designation"] = lines[i + 1].strip() if i + 1 < len(lines) else ""
            elif "Dates employed and Duration" in line:
                if i + 1 < len(lines):
                    duration_str = lines[i + 1].strip()
                    exp_info["Experience"] = extract_experience_duration(duration_str)
                exp_info["Experience"] = lines[i + 1].strip() if i + 1 < len(lines) else ""
            elif "Position location" in line:
                if i + 1 < len(lines):
                    exp_info["Location"] = lines[i + 1].strip()
        
        # Only add if designation and employer are present
        if exp_info["Designation"] and exp_info["Current Employer"]:
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
        linkedin_df (DataFrame): DataFrame containing LinkedIn profiles.
        naukri_df (DataFrame): DataFrame containing Naukri profiles.
        selected_health (str): Selected profile health level ('High', 'Medium', 'Low').
    
        linkedin_df (DataFrame): LinkedIn profiles
        naukri_df (DataFrame): Naukri profiles
        selected_health (str): Profile health filter (High, Medium, Low)
    Returns:
        DataFrame: DataFrame containing matched candidate profiles with debug information.
        DataFrame: Matched candidates
    """
    # Filter LinkedIn profiles based on selected profile health
    linkedin_filtered = linkedin_df[linkedin_df["Profile Health"] == selected_health]
    
    # Extract all experiences from LinkedIn profiles
    # If no LinkedIn profiles match the selected health, return an empty DataFrame
    if linkedin_filtered.empty:
        return pd.DataFrame()
    linkedin_filtered = linkedin_filtered.copy()
    linkedin_filtered['Experiences'] = linkedin_filtered.apply(extract_experiences, axis=1)
    
    # Preprocess names: remove punctuation, lowercase, strip spaces
    linkedin_filtered['Processed Name'] = linkedin_filtered['Candidate Name'].apply(preprocess_name)
    naukri_df['Processed Name'] = naukri_df['Candidate Name'].apply(preprocess_name)
    
    linkedin_names = linkedin_filtered['Processed Name'].tolist()
    
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
    debug_info = []
    
    for idx, naukri_row in naukri_df.iterrows():
        naukri_name_original = naukri_row["Candidate Name"]
    for _, naukri_row in naukri_df.iterrows():
        naukri_name = naukri_row["Processed Name"]
        
        if not naukri_name:
            debug_info.append(f"Naukri row {idx+1}: Empty name.")
            continue  # Skip if no name
        
        # Exact match first
        exact_matches = linkedin_filtered[linkedin_filtered['Processed Name'] == naukri_name]
        if not exact_matches.empty:
            for _, linkedin_row in exact_matches.iterrows():
                # Combine matched profiles
                combined_profile = {**linkedin_row.to_dict(), **naukri_row.to_dict()}
                matches.append(combined_profile)
                debug_info.append(f"Naukri '{naukri_name_original}' matched exactly with LinkedIn '{linkedin_row['Candidate Name']}'.")
            continue  # Move to next Naukri candidate
        
        # Fuzzy match
        matched_name, score = fuzzy_match(naukri_name, linkedin_names, threshold=85)
        if matched_name:
            matched_row = linkedin_filtered[linkedin_filtered['Processed Name'] == matched_name].iloc[0]
            combined_profile = {**matched_row.to_dict(), **naukri_row.to_dict()}
            matches.append(combined_profile)
            debug_info.append(f"Naukri '{naukri_name_original}' fuzzy matched with LinkedIn '{matched_row['Candidate Name']}' (Score: {score}).")
        else:
            debug_info.append(f"Naukri '{naukri_name_original}' did not match any LinkedIn profiles.")
    
    return pd.DataFrame(matches), debug_info
        match_result = process.extractOne(naukri_name, linkedin_names, scorer=fuzz.token_sort_ratio)
        if match_result is not None:
            match, score = match_result
            if score >= 85:
                matched_row = linkedin_filtered[linkedin_filtered["Processed Name"] == match].iloc[0]
                combined_profile = {**matched_row.to_dict(), **naukri_row.to_dict()}
                matches.append(combined_profile)
    return pd.DataFrame(matches)

def main():
    st.title("Candidate Matching App with Enhanced Fuzzy Matching")
    st.write("Upload LinkedIn and Naukri CSV files to match and shortlist candidates based on multiple criteria.")
    
    # File Uploads
    st.title("Dynamic Candidate Matching App")
    st.write("Upload LinkedIn and Naukri CSV files to match candidates dynamically.")
    linkedin_file = st.file_uploader("Upload LinkedIn CSV", type=["csv"])
    naukri_file = st.file_uploader("Upload Naukri CSV", type=["csv"])
    
    if linkedin_file and naukri_file:
        try:
            linkedin_df = pd.read_csv(linkedin_file)
            naukri_df = pd.read_csv(naukri_file)
        except Exception as e:
            st.error(f"Error reading CSV files: {e}")
            st.stop()
        
        # Verify required columns
        required_columns_linkedin = ["Candidate Name", "Profile Health"]
        required_columns_naukri = ["Candidate Name", "Current Employer", "Experience", "Current Location"]
        
        missing_columns_linkedin = [col for col in required_columns_linkedin if col not in linkedin_df.columns]
        missing_columns_naukri = [col for col in required_columns_naukri if col not in naukri_df.columns]
        
        if missing_columns_linkedin:
            st.error(f"LinkedIn CSV is missing columns: {', '.join(missing_columns_linkedin)}")
            st.stop()
        if missing_columns_naukri:
            st.error(f"Naukri CSV is missing columns: {', '.join(missing_columns_naukri)}")
            st.stop()
        
        # Display sample data
        st.subheader("Data Samples")
        st.write("**LinkedIn Data Sample:**")
        linkedin_df = pd.read_csv(linkedin_file)
        naukri_df = pd.read_csv(naukri_file)
        st.write("Sample Data")
        st.write("LinkedIn Data:")
        st.dataframe(linkedin_df.head())
        st.write("**Naukri Data Sample:**")
        st.write("Naukri Data:")
        st.dataframe(naukri_df.head())
        
        st.markdown("---")
        
        # Select Profile Health
        profile_health = st.selectbox("Select Profile Health (High, Medium, Low):", ["High", "Medium", "Low"])
        if profile_health:
            # Perform matching
            st.write(f"Matching {profile_health} profiles from LinkedIn with Naukri profiles...")
            matched_df, debug_info = match_candidates(linkedin_df, naukri_df, profile_health)
            
            if not matched_df.empty:
                st.success(f"Found {len(matched_df)} matched candidates.")
        profile_health = st.selectbox("Select Profile Health", ["High", "Medium", "Low"])
        if st.button("Match Candidates"):
            matched_df = match_candidates(linkedin_df, naukri_df, profile_health)
            if(not matched_df.empty):
                st.write("Matched Profiles:")
                st.dataframe(matched_df)
                
                # # Display debug information
                # st.subheader("Debug Information")
                # for info in debug_info:
                #     st.write(info)
                
                # Download matched profiles
                # Allow download of matched profiles
                st.download_button(
                    label="Download Matched Profiles as CSV",
                    data=matched_df.to_csv(index=False).encode('utf-8'),
                    "Download Matched Candidates",
                    data=matched_df.to_csv(index=False),
                    file_name="matched_candidates.csv",
                    mime="text/csv"
                    mime="text/csv",
                )
            
            else:
                st.warning("No matches found with fuzzy matching.")
        
                st.write("No Matched Profiles")
if __name__ == "__main__":
    main()
