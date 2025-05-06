# 🖼️ Buscador de Imagens com CLIP + ChromaDB

Um aplicativo web poderoso para categorização semântica e busca de imagens utilizando inteligência artificial de ponta, com interface amigável e sistema de autenticação seguro.

![Exemplo da Interface](D:\\Categorizador de Imagens\\Exemplo buscador de imagens.png)

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Recursos](#recursos)
- [Conceitos Técnicos](#conceitos-técnicos)
- [Instalação](#instalação)
- [Utilização](#utilização)
- [Segurança](#segurança)
- [Benefícios para Designers Gráficos](#benefícios-para-designers-gráficos)
- [Arquitetura Técnica](#arquitetura-técnica)
- [Limitações e Trabalhos Futuros](#limitações-e-trabalhos-futuros)

## 🔍 Visão Geral

Este projeto é um categorizador e buscador de imagens baseado em descrições textuais, utilizando o modelo CLIP (Contrastive Language-Image Pre-training) da OpenAI e o banco de dados vetorial ChromaDB para armazenamento persistente. O sistema é implementado como um aplicativo web interativo com Streamlit, permitindo que designers gráficos, fotógrafos e profissionais criativos organizem e localizem rapidamente imagens em suas coleções pessoais ou profissionais através de linguagem natural.

## 💎 Recursos

- **Busca Semântica**: Encontre imagens usando descrições de texto naturais
- **Interface Intuitiva**: Aplicativo web amigável construído com Streamlit
- **Processamento Eficiente**: Indexação e busca rápida usando embeddings vetoriais
- **Armazenamento Persistente**: Banco de dados vetorial ChromaDB para manter os embeddings entre sessões
- **Autenticação Segura**: Sistema de login para proteger seus dados
- **Download de Dados**: Exportação do banco de dados para uso em diferentes ambientes
- **Multi-plataforma**: Funciona em Windows, MacOS e Linux

## 🧠 Conceitos Técnicos

### Visão Computacional e IA Multimodal

O projeto utiliza conceitos avançados de visão computacional e processamento de linguagem natural:

- **CLIP (Contrastive Language-Image Pre-training)**: Modelo da OpenAI que conecta texto e imagens em um espaço vetorial comum, permitindo correspondências semânticas entre diferentes modalidades
- **Embeddings Vetoriais**: Representações numéricas de alta dimensão que codificam o conteúdo semântico de imagens e texto
- **Similaridade Coseno**: Método matemático para medir a semelhança entre vetores no espaço de embedding
- **Augmentação e Normalização de Imagens**: Técnicas de pré-processamento para melhorar a qualidade das embeddings

### Bancos de Dados Vetoriais

ChromaDB é uma tecnologia de banco de dados especializada em armazenar e consultar embeddings vetoriais:

- **Armazenamento Persistente**: Mantém embeddings entre sessões, evitando reprocessamento
- **Consultas por Similaridade**: Permite buscar os vetores mais próximos de uma consulta
- **Metadados Associados**: Armazena informações adicionais junto com os embeddings
- **Escalabilidade**: Projetado para lidar com grandes coleções de embeddings

### Transformers em Visão Computacional

O projeto utiliza a arquitetura Transformer para processamento de imagens:

- **Attention Mechanism**: Permite ao modelo focar em regiões relevantes das imagens
- **Transfer Learning**: Aproveita conhecimento pré-treinado em grandes conjuntos de dados
- **Representação Contextual**: Cria embeddings que capturam o contexto semântico completo

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- Pip (gerenciador de pacotes Python)
- CUDA (opcional, para aceleração por GPU)

### Instalação de Dependências

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/categorizador-imagens.git
cd categorizador-imagens

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Configuração da Autenticação

1. Crie a pasta `.streamlit` na raiz do projeto
2. Adicione um arquivo `secrets.toml` com suas credenciais:

```toml
# .streamlit/secrets.toml
username = "seu_usuario"
password = "sua_senha"
```

## 🖥️ Utilização

### Execução do Aplicativo

```bash
streamlit run streamlit_app.py
```

### Fluxo de Trabalho

1. **Login**: Acesse o sistema com suas credenciais
2. **Configuração**: Defina os diretórios de imagens e do banco de dados
3. **Processamento**: O sistema processará automaticamente as imagens encontradas
4. **Busca**: Use a barra lateral para buscar imagens por descrição textual
5. **Visualização**: Veja as imagens mais relevantes para sua consulta
6. **Exportação**: Baixe o banco de dados para uso em outros ambientes

## 🔒 Segurança

O sistema implementa várias medidas de segurança:

- **Autenticação por Senha**: Protege o acesso ao aplicativo
- **Comparação Segura**: Usa `hmac.compare_digest()` para evitar ataques de timing
- **Gerenciamento de Sessão**: Controle de estado da sessão do usuário
- **Limpeza de Dados Sensíveis**: Remove senhas da memória após verificação

## 🎨 Benefícios para Designers Gráficos

### Economia de Tempo e Recursos

Para designers gráficos e profissionais criativos, este sistema oferece benefícios significativos:

> "Encontre exatamente a imagem que você precisa em segundos, não em horas. Nosso Categorizador de Imagens com IA transforma a maneira como você organiza e recupera seu acervo visual. Imagine descrever a imagem que você precisa e vê-la aparecer instantaneamente, mesmo em HDs externos com milhares de arquivos. Economize tempo valioso em cada projeto, permitindo que você se concentre na criatividade, não na busca."

- **Localização Rápida**: Encontre imagens específicas em grandes coleções usando linguagem natural
- **Organização Automática**: Categorizações semânticas sem necessidade de marcação manual
- **Descoberta de Recursos**: Encontre imagens relacionadas que você pode ter esquecido
- **Economia de Tempo**: Reduza drasticamente o tempo gasto procurando ativos visuais
- **Aumento de Produtividade**: Foque mais no trabalho criativo e menos em tarefas administrativas

### Casos de Uso Específicos

- **Portfólios Extensos**: Encontre trabalhos anteriores para referência ou apresentação
- **Bibliotecas de Stock**: Organize e acesse facilmente suas coleções de imagens licenciadas
- **Arquivos de Projetos**: Localize rapidamente ativos de projetos antigos
- **Discos Externos**: Busque em HDs de backup ou armazenamento sem precisar navegar manualmente
- **Inspiração Visual**: Encontre imagens com elementos ou estilos específicos para inspiração

## 🏗️ Arquitetura Técnica

O projeto segue uma arquitetura de múltiplas camadas:

1. **Camada de Interface**: Streamlit para UI interativa
2. **Camada de Autenticação**: Sistema de login baseado em session_state
3. **Camada de Processamento**: Transformers (CLIP) para geração de embeddings
4. **Camada de Armazenamento**: ChromaDB para persistência de dados vetoriais
5. **Camada de Utilitários**: Funções auxiliares para manipulação de imagens e processamento

### Diagrama de Fluxo

```
Usuário → Autenticação → Interface Web → Processamento de Consulta → 
                                 ↑                     ↓
                     Armazenamento Persistente ← Geração de Embeddings
```

## 🔄 Limitações e Trabalhos Futuros

### Limitações Atuais

- Tempo de processamento inicial para grandes coleções de imagens
- Limitações de tamanho dependendo da memória disponível
- Suporte limitado a formatos de imagem específicos

### Desenvolvimentos Futuros

- Implementação de processamento em lote para grandes coleções
- Suporte a clustering automático para descoberta de categorias
- Interface de administração para gerenciamento de usuários
- Integração com serviços de armazenamento em nuvem
- Exportação de metadados e categorias para sistemas externos
