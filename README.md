# 🧠 GenAI Recipe Chatbot  
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)  
[![LLM](https://img.shields.io/badge/Model-OpenAI%20%7C%20HuggingFace-green)](https://huggingface.co/)  
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20Flask-orange)](https://streamlit.io/)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

> AI-powered cooking companion that generates, personalizes, and translates recipes from your ingredients.

---

## 📘 Overview

The **GenAI Recipe Chatbot** is an intelligent, multimodal cooking assistant built with **Large Language Models (LLMs)**.  
It generates creative and structured recipes, suggests alternative ingredients, and supports **voice input**, **personalized dietary filters**, and **multilingual translation**.

Developed as part of a **Generative AI course project**, this app explores prompt engineering, decoding strategies, and user experience design for recipe generation.

> **Goal:** Create an interactive recipe generator that bridges natural language understanding with real-world kitchen needs.

---

## 🧩 Core Features

### 🔊 1. Voice Input
- Speak your ingredients directly to the app.
- Uses the `speech_recognition` library with the Google Web Speech API.
- Automatically converts spoken words into ingredient text.
- Includes noise adjustment and timeout handling for smooth performance.

> Example: Say “egg” or “tomato” — each ingredient is added automatically.

### 🌱 2. Personalized Recipe Generation
- Choose from **four dietary preferences**:
  - Any
  - Low-Carb
  - High-Protein
  - Vegan
- Combine with **two complexity levels**:
  - Simple
  - Detailed
- Generates one of **eight combinations** tailored to user preferences.  
- Implemented via **prompt engineering** and dynamic control of model input text.

> Example: “Low-Carb + Simple” produces concise, lightweight dishes.

### 🌍 3. Translation Feature
- Translate generated recipes into **Chinese, French, or Spanish**.
- Built using the `googletrans` library — lightweight and fast.
- Translations are accurate and easily saved locally.

> Quick multilingual support for non-English users with near real-time speed.

---

## ⚙️ Model Experimentation

### 1. **Temperature Control**
- **Low temperature (0.5):** Coherent, stable, less creative.
- **Medium (1.0):** Balanced tone and ingredient variety.
- **High (2.0):** Creative and surprising but less consistent.  

> Higher temperature = more randomness; lower = more structure.

### 2. **Sampling Strategies**
| Parameter | Effect |
|------------|--------|
| **Top-k (5)** | Restricts output to most probable words, predictable but safe |
| **Top-p (0.7)** | Keeps cumulative probability low, coherent and focused |
| **Top-k=50, Top-p=0.95** | High diversity, creative combinations |

Combining `top_k` and `top_p` helps balance between novelty and structure.

### 3. **Decoding Strategy**
- **Greedy decoding (num_beams=1):** Faster, diverse but inconsistent.
- **Beam search (num_beams=5):** More structured and logical output.

> Beam search yields recipes with clearer formatting and more coherent steps.

---

## 🧾 Structured Recipe Formatting

The chatbot enforces a **standardized recipe structure**:

```
Title: Egg Fried Rice

Ingredients:
- Rice
- Eggs
- Green Onion

Steps:
1. Heat oil in a pan.
2. Add beaten eggs, scramble lightly.
3. Mix with cooked rice and chopped green onions.
```

- Ensures consistent readability across responses.
- Automatically adjusts **max_length** for concise (150 tokens), detailed (350), or creative recipes.

---

## 🔍 Ingredient Substitution Search

The app suggests alternative ingredients using both **direct search** and **ANN (Approximate Nearest Neighbor)** search via **Annoy**.

| Ingredient | Alternatives (Sample) | Method |
|-------------|------------------------|---------|
| Milk | Cream, Yogurt, Almond Milk | Direct Search |
| Rice | Quinoa, Couscous, Barley | ANN Search |
| Shrimp | Crab, Scallops, Lobster | Both |

- ANN search offers slightly faster results but struggles with typos (e.g., “shrimmp”).
- Both methods accurately identify semantic similarity between ingredients.

---

## 📊 Performance & Insights

- **Beam search** improved logical consistency by ~15% over greedy decoding.
- **Top-k/p tuning** enhanced recipe diversity without breaking structure.
- **Speech recognition** achieved ~90% accuracy under quiet conditions.
- **Translation** was accurate for Chinese, French, and Spanish (verified via GPT cross-check).

> The model effectively balances creativity, coherence, and speed — crucial for a real-world cooking assistant.

---

## 🧠 Tech Stack

- **Python 3.10**  
- **OpenAI / Hugging Face Transformers**  
- **SpeechRecognition + Google Web Speech API**  
- **googletrans** for multilingual translation  
- **Annoy** for ingredient similarity search  
- **Streamlit / Flask** for UI and deployment

---

## 📂 Project Structure

```
GenAI-Recipe-Chatbot/
├── app.py                     # Main chatbot application
├── recipe_generation.py       # LLM prompt and decoding control
├── ingredient_search.py       # Direct & ANN search modules
├── translation.py             # Multilingual translation feature
├── voice_input.py             # Voice-to-text ingredient input
├── utils/                     # Helper functions
└── assets/                    # Screenshots, sample outputs
```

---

## 👩‍🍳 Author

**Siyun He**  
Khoury College of Computer Sciences, Northeastern University  
📧 he.siyun@northeastern.edu  
🌐 [GitHub: HEsiyun](https://github.com/HEsiyun)

---

## 💡 Key Takeaways

- Generative AI models can **generate structured, creative recipes** with minimal human input.  
- **Prompt tuning** and **decoding parameters** directly influence coherence and creativity.  
- Multimodal interaction (voice + translation) enhances real-world usability.  

> “Cooking meets computation — where creativity gets a neural boost.”

---

## 💚 Acknowledgments

- OpenAI / Hugging Face for API and LLM support.  
- Google Web Speech API and `googletrans` for accessibility features.  
- Professor **Ryan Rad** for guidance in the Generative AI coursework.  

---

⭐ **If you enjoy this project, give it a star on GitHub!**

