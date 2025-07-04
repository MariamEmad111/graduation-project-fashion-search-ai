# ✨ Fashion Product Search Engine

An intelligent AI-powered search engine that allows users to find fashion products from **DeFacto** by providing:
- 🖼️ an image (e.g., photo or screenshot)
- 🔤 a text description (in **Arabic**, **English**, or **Arabizi**)
- 📸 or both together for more accurate results

---

## 📌 Features

- 🔎 **Multilingual Text Search**  
  Supports Arabic, English, and Arabizi (Franco-Arabic) using a smart translation and normalization pipeline.

- 🧠 **Fine-tuned mBART50 with LoRA**  
  The mBART50 model has been fine-tuned using **LoRA** (Low-Rank Adaptation) for high-quality Arabic-to-English translation, specifically in the fashion domain.

- 🧠 **CLIP-based Visual-Text Matching**  
  Uses a locally saved, pre-trained **CLIP model** to generate embeddings for both text and images, enabling efficient cross-modal similarity search.

- 🖼️ **Image Search**  
  Input a fashion image, and the system retrieves visually similar items from the dataset.

- 🔤 **Text + Image Fusion**  
  Combines both inputs to provide more contextually relevant results.

- 🧹 **Smart Preprocessing Pipeline**  
  - Language detection (Arabic, English, Arabizi)  
  - LoRA-based mBART translation for Arabic  
  - Arabizi word replacement  
  - Fashion keyword extraction using a curated vocabulary  
  - Stopword removal and token cleaning

---

## 🛠️ Tech Stack

| Component         | Description                                  |
|------------------|----------------------------------------------|
| Python           | Main development language                    |
| Hugging Face     | CLIP, mBART50, Tokenizers                    |
| PyTorch          | Model inference                              |
| LoRA             | Fine-tuning method for lightweight adaptation|
| scikit-learn     | Cosine similarity computation                |
| Pandas           | Dataset loading and manipulation             |
| NLTK             | Tokenization and stopword removal            |
| PIL              | Image processing                             |
| Langdetect       | Language detection                           |

---

## 🧪 Model Fine-Tuning

- 🧠 Fine-tuned [mBART50](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) using **LoRA**
- Adapted the model to perform better on **Arabic fashion queries**
- Merged the fine-tuned LoRA adapters with the base model
- Used this improved model during runtime to translate Arabic queries before embedding

---

## 📂 Dataset

The search engine uses a fashion dataset of **DeFacto** products, including:
- Product images
- CLIP-based image embeddings
- Product metadata
- Direct product links

> 🔗 Dataset path: `D:/graduation_project/data/defacto_final.csv`

---

## 🚀 How It Works

1. User provides input: **image**, **text**, or **both**
2. System preprocesses the input:
   - Detects language
   - Translates Arabic using fine-tuned mBART
   - Replaces Arabizi terms
   - Filters for fashion-relevant keywords
   - Embeds text/image using CLIP
3. Embeddings are compared with the dataset using **cosine similarity**
4. The top 5 most similar product links are returned

---

## 📸 Example Use Cases

- `"بلوزة بيضاء من ديفاكتو"` → top 5 matching white blouses  
- Uploading an image of a dress → visually similar dresses  
- Text + image → more accurate results with better context

---


---

## 💡 Future Improvements

- 🔄 Automate the **data scraping process** instead of relying on manual collection  
- 🛍️ Extend support to **multiple fashion brands** beyond DeFacto  
- 📈 Integrate a **ranking algorithm** based on click-through or user feedback  
- ⚡ Optimize embedding search with **ANN (Approximate Nearest Neighbor)** methods like **FAISS** or **ScaNN**  
- 📊 Add analytics to track popular queries, products, and trends  
- 🧠 Explore **multilingual fine-tuning** for mBART to cover more dialects and regional Arabic variations  
- 📷 Improve image preprocessing (e.g., background removal, cropping) for more accurate results  
- 🧪 Benchmark performance using a test set of real user queries + human evaluation  

 

 


 

