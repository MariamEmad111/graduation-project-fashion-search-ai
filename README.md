# 🧠 Fashion Search Engine with AI and Multilingual NLP

This is a graduation project focused on building a **Fashion Search Engine** that allows users to search for clothing products using **text**, **images**, or a **combination of both**. The system supports both **English** and **Arabic** queries, using powerful deep learning models to understand and match user intent with fashion items.

##  Key Features

-  **Text and Image-based Search** using the Fashion-CLIP model fine-tuned for fashion data
-  **Multilingual NLP**: Support for Arabic and English using mBART
-  **Fashion-specific Query Processing** (colors, types, materials, brands)
-  **Fashio-CLIP-based Embeddings**: Text, image, and combined embedding vectors
-  Smart query preprocessing: translation, tokenization, cleaning

---

##  Technologies Used

| Component              | Tool / Model                          |
|------------------------|----------------------------------------|
| Language Detection     | `langdetect`                           |
| Translation Model      | `mBART50` (Multilingual BART)          |
| Image-Text Embedding   | `CLIP` (Locally hosted model)          |
| Token Filtering        | Custom fashion keyword extractor       |
| NLP Preprocessing      | `nltk`, `regex`, `stopwords`           |
| Programming Language   | `Python`                               |

---

##   System Pipeline

Here's an overview of the data flow and logic:

### 1. **User Input**
- Accepts:  
  - Text query (Arabic or English)  
  - Image  
  - Both (text + image)

### 2. **Text Preprocessing Pipeline**
- Detect input language (fallback to Arabic if Arabic letters detected)
- Translate Arabic → English (using `mBART50`)
- Replace Arabizi (Arabic written with Latin letters)
- Clean, lowercase, remove stopwords
- Extract **fashion-related tokens** (colors, brands, types, etc.)

### 3. **Embedding Generation**
- If only text: Generate text embedding via CLIP
- If only image: Generate image embedding via CLIP
- If both: Combine embeddings (50% text + 50% image)
 

---
 

