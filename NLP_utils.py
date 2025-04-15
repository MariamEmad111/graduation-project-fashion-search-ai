import json
import re
import os
import pickle
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import nltk
from nltk.corpus import stopwords

# Load stopwords (only once)
nltk.download('stopwords', quiet=True)
STOPWORDS_PATH = "D:/graduation_project/data/stopwords.pkl"

if os.path.exists(STOPWORDS_PATH):
    with open(STOPWORDS_PATH, 'rb') as f:
        stop_words = pickle.load(f)
else:
    stop_words = set(stopwords.words('english'))
    with open(STOPWORDS_PATH, 'wb') as f:
        pickle.dump(stop_words, f)

# Load mBART model and tokenizer
MBART_MODEL_PATH = "D:/graduation_project/mBART"
tokenizer = MBart50Tokenizer.from_pretrained(MBART_MODEL_PATH)
model = MBartForConditionalGeneration.from_pretrained(MBART_MODEL_PATH)

# Load Arabizi dictionary
with open("D:/graduation_project/data/arabic_to_english.json", "r", encoding="utf-8") as f:
    arabizi_dict = json.load(f)

# Load fashion keywords
with open("D:/graduation_project/data/fashion_tokens.json", "r", encoding="utf-8") as f:
    fashion_data = json.load(f)
    fashion_keywords = set(
        word.lower()
        for category in ["colors", "types", "materials", "brands"]
        for word in fashion_data.get(category, [])
    )

# ====== Core Functions ======

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang != "ar" and re.search(r'[\u0600-\u06FF]', text):
            return "ar"
        return lang
    except:
        return "unknown"

def smart_translate(text: str) -> str:
    tokenizer.src_lang = "ar_AR"
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def replace_arabizi(text: str) -> str:
    for ar_word, en_word in arabizi_dict.items():
        text = re.sub(fr'\b{re.escape(ar_word)}\b', en_word, flags=re.IGNORECASE)
    return text

def clean_query(text: str) -> list:
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in stop_words]

def extract_fashion_tokens(tokens: list) -> list:
    return [t for t in tokens if t in fashion_keywords]

# ====== Public API ======

def process_query(query: str) -> str:
    """
    Takes any user query (in Arabic, English, or Arabizi),
    and returns cleaned, translated fashion-related keywords.
    """
    lang = detect_language(query)

    if lang == "ar":
        try:
            query = smart_translate(query)
        except:
            pass

    query = replace_arabizi(query)
    tokens = clean_query(query)
    relevant_tokens = extract_fashion_tokens(tokens)
    return " ".join(relevant_tokens)
