import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from generate_through_story import GenerateThroughStory
from add_monologues import AddMonologues
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_scripts(story, genre, tone, characters, temperature):
    final_script = ''
    try:
        # Split the string of characters into a list
        characters_list = [character.strip() for character in characters.split(',')]

        # Generate descriptions for each character
        characters_and_descriptions = {}
        for character in characters_list:
            character_description = GenerateThroughStory().generate_character_descriptions(story, character, temperature)
            characters_and_descriptions[character] = character_description

        # Convert the dictionary into a list of dictionaries
        character_descriptions_list = [{"character": character, "description": description} for character, description in characters_and_descriptions.items()]

        chapter = GenerateThroughStory().generate_heading_and_descriptions(story, genre, tone, temperature)
        chapter_description = chapter[0]["description"]
        
        dialogues = GenerateThroughStory().generate_dialogues(story, chapter_description, characters_list, character_descriptions_list, genre, tone, temperature)
        elements = AddMonologues().identify_emotions_and_dialogues(dialogues)
        inner_monologues = AddMonologues().add_inner_monologues(elements, chapter_description, temperature)
        final_script = AddMonologues().final_script(elements, inner_monologues)
    except Exception as e: 
        print(e)

    return final_script

# Preprocess text
def preprocess_text(text):
    return text.lower()  #Convert text to lowercase

# Compute cosine similarity
def compute_similarity(reference, candidate):
    # Preprocess text
    reference = preprocess_text(reference)
    candidate = preprocess_text(candidate)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform text
    tfidf_matrix = vectorizer.fit_transform([reference, candidate])
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0,1]
    
    return similarity_score

def calculate_lsa_similarity(original_story, generated_script):
    # Preprocess text
    original_story = preprocess_text(original_story)
    generated_script = preprocess_text(generated_script)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform vectorizer on original story and generated script
    tfidf_matrix = vectorizer.fit_transform([original_story, generated_script])
    
    # Check the number of features (terms) in the document-term matrix
    n_features = tfidf_matrix.shape[1]
    
    # Adjust the number of components for LSA based on the number of features
    n_components = min(100, n_features)  # Set the maximum number of components
    
    # Apply LSA (Truncated SVD)
    lsa = TruncatedSVD(n_components=n_components)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    
    # Calculate cosine similarity between LSA representations
    similarity_score = cosine_similarity(lsa_matrix[0:1], lsa_matrix[1:2])[0][0]
    
    return similarity_score

def evaluate_script(reference, candidate):
    similarity_score = compute_similarity(reference, candidate)
    lsa_similarity = calculate_lsa_similarity(reference, candidate)
    return similarity_score, lsa_similarity

def generate_and_store_scripts(input_data, temperatures):
    for temp in temperatures:
        temp_dataset = pd.DataFrame(columns=['story', 'genre', 'tone', 'characters', 'script', 'temperature', 'similarity,' 'las similarity'])
        for _, row in input_data.iterrows(): #Ignoring the index value
            # Extract data from the current row
            story = row['story']
            genre = row['genre']
            tone = row['tone']
            characters = row['characters']
            
            # Generate script using the generate_scripts function
            script = generate_scripts(story, genre, tone, characters, temp)
            
            # Evaluate the generated script
            similarity_score, lsa_similarity = evaluate_script(story, script)
            rows_to_append = [{'story': story,
                   'genre': genre,
                   'tone': tone,
                   'characters': characters,
                   'script': script,
                   'temperature': temp,
                   'similarity': similarity_score,
                   'lsa similarity' : lsa_similarity}]

            temp_dataset = pd.concat([temp_dataset, pd.DataFrame(rows_to_append)], ignore_index=True)

            # Get the directory of the current script and store the new dataset for the current temperature
            temp_dataset.to_csv(os.path.join(os.path.dirname(__file__), '..', 'datasets', f'dataset_temp_{temp}_test.csv'), index=False)

        print(f"Dataset for {temp} is done!")


# Get the directory of the current script
current_directory = os.path.dirname(__file__)

# Construct the path to the original dataset CSV file 
csv_file_path = os.path.join(current_directory, '..', 'datasets', 'input_values.csv')

# Read the CSV file
input_data = pd.read_csv(csv_file_path)

# Define the range of temperatures to generate scripts for
temperatures = [0.3, 0.4, 0.5, 0.6, 0.7]

# Generate and store scripts for each temperature
generate_and_store_scripts(input_data, temperatures)

