#%%
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from chromadb import PersistentClient

# Configuração inicial
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregar modelo CLIP (versão maior)
model_name = 'openai/clip-vit-large-patch14'
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.to(device)

# Diretório das imagens
IMAGE_DIR = "Imagens"  # Substitua pelo caminho real
CHROMA_DB_PATH = "./chroma_db"  # Caminho para armazenar o banco de dados ChromaDB

# Função para aplicar augmentations leves (brilho e contraste)
def augment_image(image):
    # Converter para RGB se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Ajustar brilho (fator entre 0.8 e 1.2)
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(1.1)  # Aumenta ligeiramente o brilho
    
    # Ajustar contraste (fator entre 0.9 e 1.1)
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(1.1)  # Aumenta ligeiramente o contraste
    
    return image

# Função para normalizar imagem
def normalize_image(image):
    # Redimensionar para o tamanho esperado pelo CLIP (geralmente 224x224 para large-patch14)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Converter para array e normalizar (média e desvio padrão do ImageNet)
    transform = lambda x: (np.array(x) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    normalized = transform(image)
    return Image.fromarray((normalized * 255).astype(np.uint8))

# Função para gerar embedding de imagem (com augmentations e normalização)
def get_image_embedding(image_path):
    try:
        # Carregar imagem
        image = Image.open(image_path).convert('RGB')
        
        # Aplicar augmentations
        image = augment_image(image)
        
        # Normalizar
        image = normalize_image(image)
        
        # Processar com CLIP
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalizar o embedding
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return None

# Função para gerar embedding de texto
def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().flatten()

# Inicializar ou carregar ChromaDB
def initialize_chromadb():
    client = PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name="image_embeddings")
    except:
        collection = client.create_collection(name="image_embeddings")
    return collection

# Processar e armazenar embeddings de imagens
def process_images(image_dir, collection):
    print(f"Contagem atual de documentos no ChromaDB: {collection.count()}")
    if collection.count() > 0:
        docs = collection.get()["ids"]
        print(f"IDs já processados: {docs}")
        processed_images = set(docs)
    else:
        processed_images = set()

    print(f"Imagens já processadas: {processed_images}")
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')) and img_name not in processed_images:
            image_path = os.path.join(image_dir, img_name)
            embedding = get_image_embedding(image_path)
            if embedding is not None:
                collection.add(
                    ids=[img_name],
                    embeddings=[embedding.tolist()],  # ChromaDB espera lista
                    metadatas=[{"path": image_path}]
                )
                print(f"Processada e armazenada: {img_name}")
        else:
            print(f"Ignorada (já processada): {img_name}")

# Função para buscar imagens mais similares
def find_most_similar_images(query, collection, n_results=2):  # Mantido em 2 para plotar apenas duas
    query_embedding = get_text_embedding(query)
    
    # Buscar no ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    return results["ids"][0], results["distances"][0]  # IDs das imagens e suas distâncias

# Teste interativo com plotagem
if __name__ == "__main__":
    # Inicializar ChromaDB
    collection = initialize_chromadb()

    # Processar imagens não processadas
    process_images(IMAGE_DIR, collection)

    # Loop interativo para testes
    while True:
        query = input("Digite uma descrição (ex.: 'praia ensolarada') ou 'sair' para encerrar: ")
        if query.lower() == 'sair':
            break

        # Buscar imagens similares
        image_ids, distances = find_most_similar_images(query, collection)
        
        # Mostrar resultados e plotar
        print("\nImagens mais similares:")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Criar figura com 2 subplots

        for idx, (img_id, dist) in enumerate(zip(image_ids, distances)):
            print(f"Imagem {idx+1}: {img_id}, Distância: {dist:.4f} (menor distância = mais similar)")
            image_path = collection.get(ids=[img_id])["metadatas"][0]["path"]
            
            # Carregar e plotar a imagem
            img = mpimg.imread(image_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Imagem: {img_id}\nSimilaridade: {dist:.4f}")
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

        # Opcional: imprimir caminhos
        for img_id in image_ids:
            image_path = collection.get(ids=[img_id])["metadatas"][0]["path"]
            print(f"Caminho da imagem: {image_path}")
# %%
