import streamlit as st
import pandas as pd
import spacy
import os
import gdown
from annoy import AnnoyIndex
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator
import speech_recognition as sr


# ✅ Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model 'en_core_web_lg'...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"

@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):  
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    return pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()

ingredient_list = load_ingredient_data()

# ✅ Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []

    for ing in ingredient_list:
        vec = nlp(ing.lower()).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), filtered_ingredients

ingredient_vectors, filtered_ingredient_list = compute_embeddings()

# ✅ Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = AnnoyIndex(dim, metric="angular")  # ✅ Uses angular distance (1 - cosine similarity)
    
    for i, vec in enumerate(ingredient_vectors):
        index.add_item(i, vec)
    
    index.build(50)  # ✅ More trees = better accuracy
    return index
annoy_index = build_annoy_index()

# ✅ Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.any(vec1) and np.any(vec2) else 0

def direct_search_alternatives(ingredient):
    input_vector = nlp(ingredient.lower()).vector

    similarities = []

    # Compute cosine similarity between the input ingredient and all other ingredients
    for i, ing in enumerate(filtered_ingredient_list):
        if ing.lower() == ingredient.lower():
            continue  # Skip the input ingredient itself

        ing_vector = ingredient_vectors[i]
        similarity = cosine_similarity(input_vector, ing_vector)
        similarities.append((ing, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    top_3_alternatives = [ing for ing, _ in similarities[:3]]

    return top_3_alternatives


# ✅ Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient):
    index = build_annoy_index()
    input_vector = nlp(ingredient.lower()).vector
    nearest_indices = index.get_nns_by_vector(input_vector, 4, include_distances=False)

    # Filter out the input ingredient from the alternatives
    top_3_alternatives = [
        filtered_ingredient_list[i] for i in nearest_indices
        if filtered_ingredient_list[i].lower() != ingredient.lower()
    ][:3]

    return top_3_alternatives


# ✅ Generate Recipe
def generate_recipe(ingredients, cuisine, complexity, num_beams):
    input_text = (
        f"Generate a {complexity} recipe.\n"
        f"Then list the ingredients and provide step-by-step instructions.\n"
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
    )    
    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], 
                             max_length=1000, 
                             num_return_sequences=1, 
                             num_beams=num_beams,
                             do_sample=True,
                             repetition_penalty=1.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

def generate_recipe_low_carb(ingredients, cuisine, complexity, num_beams):
    input_text = (
        f"Generate a {complexity} low-carb recipe with comprehensive instructions.\n"
        f"Ensure the recipe is low in carbohydrates.\n"
        f"Provide step-by-step instructions.\n"
        f"Ingredients: {', '.join(ingredients.split(', '))}.\n"
        f"Cuisine: {cuisine}.\n"
    )
    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"],
                             max_length=1000,
                             num_return_sequences=1,
                             num_beams=num_beams,
                             do_sample=True,
                             repetition_penalty=1.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()


def generate_high_protein_recipe(ingredients, cuisine, complexity, num_beams):
    input_text = (
        f"Generate a {complexity} high-protein recipe with comprehensive instructions.\n"
        f"Ensure the recipe is high in protein.\n"
        f"Provide step-by-step instructions.\n"
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
    )

    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"],
                             max_length=1000,
                             num_return_sequences=1,
                             num_beams=num_beams,
                             do_sample=True,
                             repetition_penalty=1.2)

    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

def generate_vegan_recipe(ingredients, cuisine, complexity,num_beams):
    input_text = (
        f"Generate a {complexity} vegan recipe with comprehensive instructions.\n"
        f"Ensure the recipe is entirely plant-based and free from animal products.\n"
        f"Provide step-by-step instructions.\n"
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
    )

    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"],
                             max_length=1000,
                             num_return_sequences=1,
                             num_beams=num_beams,
                             do_sample=True,
                             repetition_penalty=1.2)

    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

# ✅ Translate Recipe
def translate_recipe(recipe, target_language):
    translator = Translator()
    translation = translator.translate(recipe, dest=target_language)
    return translation.text


# ✅ Function to capture voice input
def get_voice_input(timeout=5, phrase_time_limit=10):
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("Listening for ingredients... Please speak clearly.")
        
        # Adjust for ambient noise (helps in noisy environments)
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            # Listen with timeout to avoid waiting indefinitely
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Try speaking again.")
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please speak more clearly.")
        except sr.RequestError:
            st.error("Could not request results; check your internet connection.")
    
    return None

# ✅ Streamlit App UI
# Initialize session state for ingredients if not already done
if "ingredients" not in st.session_state:
    st.session_state["ingredients"] = []

st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")

# Add a button for voice input
if st.button("🎤 Use Voice Input for Ingredients"):
    voice_input = get_voice_input()
    if voice_input:
        # Append the recognized ingredient to the session state list
        st.session_state["ingredients"].append(voice_input)
st.markdown(
    """
    <div style="
        padding: 10px; 
        background-color: #f9f5ff; 
        border-radius: 10px; 
        color: #6a0dad; 
        font-size: 18px; 
        font-weight: bold; 
        text-align: center;">
        🔊 Please say **one word at a time** when using voice input.
    </div>
    """, 
    unsafe_allow_html=True
)

# Display the current list of ingredients
st.write("Current Ingredients:", ", ".join(st.session_state["ingredients"]))

# Use the voice input if available, otherwise fall back to text input
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):", value=", ".join(st.session_state["ingredients"]))

cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])
dietary = st.selectbox("dietary preferences:", ["Any", "low-carb", "high-protein", "vegan"])
complexity = st.selectbox("Select Recipe Complexity:", ["Simple", "Detailed"])
if st.button("Generate Recipe", use_container_width=True) and ingredients:
    if dietary == "Any":
        if complexity == "Simple":
            st.session_state["recipe"] = generate_recipe(ingredients, cuisine, "brief", num_beams=5)
        else:
            st.session_state["recipe"] = generate_recipe(ingredients, cuisine, "detailed", num_beams=5)
    elif dietary == "low-carb":
        if complexity == "Simple":
            st.session_state["recipe"] = generate_recipe_low_carb(ingredients, cuisine, "brief", num_beams=5)
        else:
            st.session_state["recipe"] = generate_recipe_low_carb(ingredients, cuisine, "detailed", num_beams=5)
    elif dietary == "high-protein":
        if complexity == "Simple":
            st.session_state["recipe"] = generate_high_protein_recipe(ingredients, cuisine, "brief", num_beams=5)
        else:
            st.session_state["recipe"] = generate_high_protein_recipe(ingredients, cuisine, "detailed", num_beams=5)
    else:
        if complexity == "Simple":
            st.session_state["recipe"] = generate_vegan_recipe(ingredients, cuisine, "brief", num_beams=5)
        else:
            st.session_state["recipe"] = generate_vegan_recipe(ingredients, cuisine, "detailed", num_beams=5)


if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    st.download_button(label="📂 Save Recipe", 
                       data=st.session_state["recipe"], 
                       file_name="recipe.txt", 
                       mime="text/plain")
    
    # ✅ Translation Section
    st.markdown("---")
    st.markdown("## 🌐 Translate Recipe")
    language = st.selectbox("Select a language for translation:", ["Chinese", "French", "Spanish"])
    language_codes = {"Chinese": "zh-cn", "French": "fr", "Spanish": "es"}

    if st.button("Translate Recipe", use_container_width=True):
        st.session_state["translated_recipe"] = translate_recipe(st.session_state["recipe"], language_codes[language])
        st.markdown(f"### 🌍 Translated Recipe in {language}:")
        st.text_area("Translated Recipe:", st.session_state["translated_recipe"], height=200)
    
    if "translated_recipe" in st.session_state:
        st.download_button(label="📂 Save Translated Recipe",
                           data=st.session_state["translated_recipe"],
                           file_name="translated_recipe.txt",
                           mime="text/plain")
        
    # ✅ Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio("Select Search Method:", ["Annoy (Fastest)", "Direct Search (Best Accuracy)"], index=0)

    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives
        }
        alternatives = search_methods[search_method](ingredient_to_replace)
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
