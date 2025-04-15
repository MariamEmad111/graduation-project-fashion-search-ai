import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from typing import Union, List, Tuple

# ========== 1. Load Model, Tokenizer, and Processor ==========

def load_model_with_tokenizer(model_path: str) -> Tuple[CLIPProcessor, CLIPModel, CLIPTokenizer]:
    """
    Load the CLIP model, processor, and tokenizer.
    
    Parameters:
        model_path (str): Path to the local directory containing the model files.

    Note:
        Ensure the model_path contains:
        - config.json
        - pytorch_model.bin
        - preprocessor_config.json
        - tokenizer_config.json
        - tokenizer.json
        - vocab.json
        - special_tokens_map.json
    """
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)

    # Update the paths below to point to your local tokenizer files
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path,
        special_tokens_map=os.path.join(model_path, "special_tokens_map.json"),
        tokenizer_file=os.path.join(model_path, "tokenizer.json"),
        tokenizer_config=os.path.join(model_path, "tokenizer_config.json"),
        vocab_file=os.path.join(model_path, "vocab.json")
    )

    return processor, model, tokenizer

# ========== 2. Get Text Embeddings ==========

def get_text_embedding(texts: Union[str, List[str]], processor, model, tokenizer) -> List:
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=77)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs.cpu().tolist()

# ========== 3. Get Image Embeddings ==========

def get_image_embedding(images: Union[str, List[str]], processor, model) -> List:
    if isinstance(images, str):
        images = [images]

    image_list = []
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
            image_list.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    if not image_list:
        raise ValueError("No valid images found.")

    inputs = processor(images=image_list, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.cpu().tolist()

# ========== 4. Combined Embeddings ==========

def get_combined_embedding(
    text: Union[str, List[str], None],
    image: Union[str, List[str], None],
    processor,
    model,
    tokenizer
) -> List:
    if text and image:
        if isinstance(text, str):
            text = [text]
        if isinstance(image, str):
            image = [image]

        image_list = []
        for p in image:
            try:
                image_list.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Error loading image {p}: {e}")

        if not image_list:
            raise ValueError("No valid images found.")

        text_inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        image_inputs = processor(images=image_list, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            image_features = model.get_image_features(**image_inputs)
            combined_features = 0.5 * text_features + 0.5 * image_features

        return combined_features.cpu().tolist()

    elif text:
        return get_text_embedding(text, processor, model, tokenizer)

    elif image:
        return get_image_embedding(image, processor, model)

    else:
        raise ValueError("Input must contain text or image or both.")
