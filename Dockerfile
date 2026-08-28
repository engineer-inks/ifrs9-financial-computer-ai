# Imagem base com PyTorch e CUDA
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Variáveis de ambiente para otimizar o Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Instalação de dependências do sistema operacional
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Configura o diretório de trabalho dentro do contêiner
WORKDIR /workspace

COPY requirements.txt .
# Atualiza o pip e instala as dependências
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia os arquivos de configuração do pacote
COPY pyproject.toml README.md ./
COPY src/ src/

# Instala o nosso pacote em modo editável (-e)
RUN pip install -e .[dev]

# O Render injeta dinamicamente a variável de ambiente $PORT.
# O comando abaixo garante que o Uvicorn vai rodar na porta correta, seja no Render ou na sua máquina local (onde o padrão será 8000).
CMD ["sh", "-c", "uvicorn src.ifrs9_framework.api:app --host 0.0.0.0 --port ${PORT:-8000}"]