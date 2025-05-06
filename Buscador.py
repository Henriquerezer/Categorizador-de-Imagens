__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# streamlit_app.py
import os
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageEnhance
import numpy as np
from chromadb import PersistentClient
import zipfile
import shutil
import tempfile
import hmac

# Inicialização das variáveis de session_state
def init_session_state():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "login_form_submitted" not in st.session_state:
        st.session_state["login_form_submitted"] = False

# Função de callback para o formulário de login
def login_callback():
    st.session_state["login_form_submitted"] = True
    username = st.session_state["username"]
    password = st.session_state["password"]
    
    # Verificar credenciais
    if username == st.secrets["username"] and hmac.compare_digest(password, st.secrets["password"]):
        st.session_state["password_correct"] = True
        # Limpar senha da session_state por segurança
        st.session_state["password"] = ""
    else:
        st.session_state["password_correct"] = False

# Função para verificar senha
def check_password():
    """Retorna `True` se o usuário informou a senha correta."""
    # Inicializar session_state
    init_session_state()
    
    # Se já estiver autenticado, retorna True
    if st.session_state["password_correct"]:
        return True
        
    # Caso contrário, mostrar tela de login
    st.title("Acesso Restrito 🔒")
    
    with st.form("login_form", clear_on_submit=False):
        st.text_input("Usuário", key="username")
        st.text_input("Senha", type="password", key="password")
        submitted = st.form_submit_button("Entrar", on_click=login_callback)
    
    # Mostrar mensagens de erro após tentativa de login
    if st.session_state["login_form_submitted"] and not st.session_state["password_correct"]:
        st.error("😕 Usuário ou senha incorretos")
        
    return st.session_state["password_correct"]

# Funções utilitárias
def augment_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ImageEnhance.Brightness(image).enhance(1.1)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    return image

def normalize_image(image):
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    transform = lambda x: (np.array(x) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    normalized = transform(image)
    return Image.fromarray((normalized * 255).astype(np.uint8))

def get_image_embedding(image_path, model, processor, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image = augment_image(image)
        image = normalize_image(image)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        st.warning(f"Erro ao processar {image_path}: {e}")
        return None

def get_text_embedding(text, model, processor, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().flatten()

def initialize_chromadb(path):
    client = PersistentClient(path=path)
    try:
        collection = client.get_collection(name="image_embeddings")
    except:
        collection = client.create_collection(name="image_embeddings")
    return collection

def process_images(image_dir, collection, model, processor, device):
    total_images = 0
    new_embeddings = 0
    ignored_images = 0
    if collection.count() > 0:
        processed_images = set(collection.get()["ids"])
    else:
        processed_images = set()
    image_list = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_list)

    progress_bar = st.progress(0)
    for idx, img_name in enumerate(image_list):
        if img_name not in processed_images:
            image_path = os.path.join(image_dir, img_name)
            embedding = get_image_embedding(image_path, model, processor, device)
            if embedding is not None:
                collection.add(
                    ids=[img_name],
                    embeddings=[embedding.tolist()],
                    metadatas=[{"path": image_path}]
                )
                new_embeddings += 1
        else:
            ignored_images += 1
        progress_bar.progress((idx + 1) / total_images)

    return total_images, new_embeddings, ignored_images

def find_most_similar_images(query, collection, model, processor, device, n_results=4):
    query_embedding = get_text_embedding(query, model, processor, device)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results["ids"][0], results["metadatas"][0], results["distances"][0]

# Função para exibir a interface principal
def main_app():
    st.title("🖼️ Buscador de Imagens com CLIP + ChromaDB")
    
    # Adicionar botão de logout e informações do usuário na barra lateral
    st.sidebar.success(f"Logado como: {st.secrets['username']}")
    if st.sidebar.button("Logout"):
        # Resetar todas as variáveis de sessão relacionadas à autenticação
        st.session_state["password_correct"] = False
        st.session_state["login_form_submitted"] = False
        st.session_state["username"] = ""
        if "password" in st.session_state:
            st.session_state["password"] = ""
        st.rerun()

    # Entrada de pastas
    image_dir = st.text_input("📁 Caminho da pasta de imagens:", value=".\Imagens")
    chroma_dir = st.text_input("📁 Caminho para o ChromaDB:", value=".\chroma_db")

    # Botão para confirmar caminhos
    if st.button("Confirmar caminhos"):
        st.session_state['confirmed'] = True

    if 'confirmed' in st.session_state:
        # Carregar modelo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'openai/clip-vit-large-patch14'
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.to(device)

        st.success(f"Modelo carregado ({device})")

        # Inicializar ChromaDB
        collection = initialize_chromadb(chroma_dir)

        st.info("⏳ Processando imagens...")
        total, new_embeds, ignored = process_images(image_dir, collection, model, processor, device)

        st.success(f"Imagens processadas! Total: {total}, Novos embeddings: {new_embeds}, Ignoradas: {ignored}")

        # Barra lateral para buscar imagens
        st.sidebar.title("🔍 Buscar imagens semelhantes")
        query = st.sidebar.text_input("Digite a descrição:")
        n_results = st.sidebar.slider("Número de imagens para retornar:", min_value=1, max_value=10, value=4)

        if st.sidebar.button("Buscar"):
            with st.spinner("Buscando imagens semelhantes..."):
                image_ids, metadatas, distances = find_most_similar_images(query, collection, model, processor, device, n_results=n_results)
                st.subheader(f"Resultados para: `{query}`")

                cols = st.columns(n_results)
                for idx, (img_id, metadata, dist) in enumerate(zip(image_ids, metadatas, distances)):
                    with cols[idx]:
                        img = Image.open(metadata["path"])
                        st.image(img, caption=f"{metadata['path']}\nSimilaridade: {dist:.4f}", use_container_width=True)

        # Botão para download do banco ChromaDB
        st.sidebar.title("💾 Baixar banco ChromaDB")
        if st.sidebar.button("Compactar e baixar"):
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                for root, dirs, files in os.walk(chroma_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file), chroma_dir))
            st.sidebar.download_button(
                label="📥 Baixar banco",
                data=open(temp_zip.name, "rb"),
                file_name="chroma_db.zip",
                mime="application/zip"
            )

# Configuração da página
st.set_page_config(page_title="Buscador de Imagens", page_icon="🖼️", layout="wide")

# Verificar autenticação e renderizar o app
if check_password():
    # Se autenticado, limpar tela e mostrar o app principal
    st.empty()
    main_app()
