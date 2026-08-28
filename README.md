# 🚀 IFRS 9 Financial Computer AI & MLOps Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange?style=for-the-badge&logo=databricks&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

*Plataforma End-to-End Self-Service de MLOps para Modelagem de Risco de Crédito (IFRS 9).*

</div>

---

## 📌 Visão Geral da Arquitetura

Este repositório implementa uma arquitetura robusta para ingestão, engenharia de features, otimização de hiperparâmetros (AutoML via Grid Search/Optuna) e auditoria regulatória de modelos de Probabilidade de Default (PD) sob as normas do **IFRS 9**.

A solução desacopla o motor matemático da interface gráfica através de uma abordagem orientada a serviços (*Microservices & SPA Monolithic Frontend*), garantindo governança, reprodutibilidade e facilidade de uso para cientistas de dados e analistas de risco.

---

## 📂 Estrutura de Diretórios

Para manter os princípios de código limpo e separação de responsabilidades, a árvore do projeto está organizada da seguinte forma:

```text
ifrs9-financial-computer-ai/
│
├── src/                      # Núcleo de Machine Learning (features, models, orchestrator)
├── config/                   # Configurações de ambiente (.env e config.yaml)
│
├── ui/                       # Camada de Apresentação (Frontend SPA)
│   ├── config_ui.html        # Painel Self-Service de Configuração e Feature Selection
│   ├── vertex_pipeline.html  # Visualizador de Dataflow e Logs de Execução
│   └── dashboard_results.html# Dashboard Executivo (Curva ROC, Matriz de Confusão, Hosmer-Lemeshow)
│
├── docker-compose.yml        # Orquestração de containers
└── Dockerfile                # Imagem oficial da aplicação FastAPI + ML
```

⚙️ Abordagem Tecnológica e Design System

1. Por que uma Interface Monolítica (SPA via CDN)?
Optamos por utilizar HTML puro estruturado com Tailwind CSS (via CDN) e JavaScript embutido. Para um painel interno de engenharia de risco em ambientes bancários, essa abordagem elimina a complexidade de build de frameworks pesados (como React ou Angular), mantendo o arquivo leve, rápido e auditável diretamente em qualquer navegador.

1. O Fluxo de Integração (Frontend ⇄ Backend)
Como navegadores possuem restrições de segurança que impedem o acesso direto de escrita em arquivos locais do servidor, implementamos uma API Ponte (FastAPI):

Ação do Usuário: O analista interage com o painel visual (config_ui.html), seleciona features e define o algoritmo (CatBoost, LightGBM, EBM).

Requisição HTTP (POST): O frontend dispara os parâmetros estruturados para a API em Python.

Persistência via IaC (ML as Code): O micro-servidor FastAPI recebe o payload e atualiza dinamicamente o arquivo config.yaml no disco.

Gatilho de Orquestração: Imediatamente após a persistência, o backend aciona o pipeline_orchestrator.py para iniciar o ciclo de treino.

📊 Governança e Armazenamento (Storage Strategy)
Gerenciamento de Configuração: O próprio arquivo yaml atua como a fonte da verdade (Single Source of Truth), garantindo rastreabilidade de infraestrutura como código.

Histórico de Experimentos: O rastreamento de métricas e artefatos de modelos é delegado nativamente ao MLflow (mlflow.db), integrado ao ecossistema Docker da aplicação.

Metrics Store: Os relatórios de performance final (incluindo curvas ROC, importância de variáveis e testes de aderência Hosmer-Lemeshow) são exportados de forma estática para JSON, alimentando o dashboard executivo.

🚀 Próximos Passos de Engenharia
[x] Conexão e preview de datasets de portfólio em formato .parquet.

[x] Painel de arrastar e soltar (Drag & Drop) para Feature Engineering.

[ ] Conclusão do microsserviço api.py utilizando FastAPI para fechar o ciclo de escrita automática do arquivo de configuração.

[ ] Expansão dos testes de estresse para validação cruzada Out-of-Time (OOT).

✒️ Autor
Eneas R. S. Junior
Eng. Machine Learning & Data Science
Desenvolvido com foco em Engenharia de Dados, Machine Learning e Arquitetura MLOps.