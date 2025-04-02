# Categorizador de Imagens com CLIP e ChromaDB

## Visão Geral
Este projeto tem como objetivo o desenvolvimento de um categorizador de imagens utilizando técnicas avançadas de Visão Computacional e Aprendizado de Máquina. A proposta é criar um sistema capaz de gerar embeddings de imagens e compará-los com embeddings textuais utilizando o modelo **CLIP** (Contrastive Language-Image Pretraining) da OpenAI. 

Atualmente, o projeto está em desenvolvimento e visa ser disponibilizado como uma aplicação interativa utilizando **Streamlit**.

## Tecnologias Utilizadas
- **Transformers (CLIP)**: Para extração de embeddings de imagens e textos.
  
**Sobre CLIP (Contrastive Language-Image Pretraining)**
O CLIP é um modelo desenvolvido pela OpenAI que treina em um grande conjunto de pares de imagens e descrições textuais, permitindo associar imagens e textos de forma eficiente. O modelo é baseado em aprendizado contrastivo, onde ele aprende a maximizar a similaridade entre descrições corretas e suas respectivas imagens, enquanto minimiza a similaridade com descrições incorretas.

   - O CLIP pode converter imagens e textos em um espaço vetorial comum, permitindo comparações diretas.

   - Ele utiliza arquiteturas como ViT (Vision Transformer) para processar imagens e Transformers para processar textos.

   - Não exige re-treinamento para novas categorias; em vez disso, ele pode classificar imagens com base em descrições fornecidas no momento da consulta.

- **Torch (PyTorch)**: Backend para inferência do modelo.
- **Pillow (PIL)**: Manipulação e pré-processamento de imagens.
- **ChromaDB**: Armazenamento e busca eficiente de embeddings.
  
**Sobre ChromaDB**
O ChromaDB é um banco de dados especializado no armazenamento e recuperação de vetores embeddeds. Ele é otimizado para buscas aproximadas em espaços vetoriais de alta dimensão, tornando-o ideal para aplicações de busca por similaridade.
No contexto deste projeto, ChromaDB é utilizado para armazenar embeddings gerados pelo CLIP, permitindo:

   - Busca eficiente por imagens semelhantes com base em consultas textuais.

   - Escalabilidade, pois pode armazenar grandes quantidades de embeddings sem comprometer o desempenho.

   - Suporte para diversos formatos de indexação, como HNSW (Hierarchical Navigable Small World), que melhora a recuperação rápida de vizinhos próximos.
- **Matplotlib**: Visualização das imagens recuperadas.

## Estrutura do Projeto
O projeto segue a seguinte estrutura:
```
├── Imagens/             # Diretório onde as imagens devem ser colocadas
├── chroma_db/           # Banco de dados para armazenamento dos embeddings
├── categorizador.py      # Script principal para processamento e consulta
└── README.md            # Este arquivo
```

## Funcionamento
1. O sistema carrega o modelo **CLIP** e processa as imagens armazenadas na pasta `Imagens/`.
2. Para cada imagem, um embedding vetorial é gerado e armazenado no banco **ChromaDB**.
3. Quando um usuário insere um texto descritivo, o sistema gera um embedding textual e busca no banco de dados as imagens mais similares.
4. As imagens mais próximas são retornadas e exibidas ao usuário.

## Trechos de Código
### Configuração Inicial
```python
import torch
from transformers import CLIPProcessor, CLIPModel

# Configuração do dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregar modelo CLIP
model_name = 'openai/clip-vit-large-patch14'
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
```

### Gerar Embedding de Imagens
```python
def get_image_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()
```

### Busca de Imagens Semelhantes
```python
def find_most_similar_images(query, collection, n_results=2):
    query_embedding = get_text_embedding(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
    return results["ids"][0], results["distances"][0]
```

## Execução
1. **Coloque suas imagens na pasta `Imagens/`**.
2. **Execute o script principal**:
   ```bash
   python categorizador.py
   ```
3. O script processará automaticamente as imagens e armazenará os embeddings no banco **ChromaDB** (`chroma_db/`).
4. Quando solicitado, digite uma descrição textual para buscar imagens semelhantes.

## Próximos Passos
- Criar interface interativa com **Streamlit**.
- Aprimorar técnicas de pré-processamento e normalização de imagens.
- Implementar suporte para múltiplas consultas simultâneas.

Este projeto está em fase de desenvolvimento e contribuições são bem-vindas!

