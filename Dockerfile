# Imagem base otimizada com PyTorch e CUDA instalados
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
# Atualiza o pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia os arquivos de configuração do pacote primeiro (aproveita o cache do Docker)
COPY pyproject.toml README.md ./
COPY src/ src/

# Instala o nosso pacote em modo editável (-e) com as dependências de desenvolvimento
# Isso significa que qualquer alteração no código na sua máquina reflete instantaneamente no contêiner
RUN pip install -e .[dev]

# Expõe as portas que usaremos para o Jupyter (8888) e MLflow (5000)
EXPOSE 8888 5000

# Comando padrão ao iniciar o contêiner (inicia o bash)
CMD ["bash"]