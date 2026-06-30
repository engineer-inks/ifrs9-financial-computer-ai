from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yaml
import subprocess
import os
import json
import logging

# Configuração simples de log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLOps-API")

app = FastAPI(title="IFRS9 MLOps API")

# Configurando CORS para permitir que o HTML comunique com a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineConfig(BaseModel):
    pipeline_name: str
    target: str
    numeric_features: list
    categorical_features: list
    yeo_johnson_features: list
    algorithm: str
    auto_tune: bool

# Caminhos absolutos/relativos baseados na pasta src/ifrs9_framework
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config/config.yaml")
METRICS_PATH = os.path.join(BASE_DIR, "config/metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data/raw/synthetic_credit_data.parquet")

# ---------------------------------------------------------
# [NOVO] PASSO 2: Rota Dinâmica de Estatísticas de Features
# ---------------------------------------------------------
@app.get("/api/feature-stats/{feature_name}")
async def get_feature_stats(feature_name: str):
    """
    Lê o ficheiro Parquet e calcula a distribuição e o risco (Default Rate)
    para a feature solicitada, devolvendo os dados prontos para o Chart.js
    """
    try:
        # 1. Carregar os dados (Apenas as colunas necessárias para poupar memória)
        target_col = "default_flag"
        
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=404, detail="Ficheiro de dados não encontrado.")
            
        df = pd.read_parquet(DATA_PATH, columns=[feature_name, target_col])
        
        # 2. Lógica de Agrupamento
        if pd.api.types.is_numeric_dtype(df[feature_name]) and df[feature_name].nunique() > 10:
            # Se for numérica contínua, partimos em 5 faixas (Quintis)
            # Usamos qcut para garantir volumes equilibrados em cada barra do gráfico
            faixas = pd.qcut(df[feature_name], q=5, duplicates='drop')
            stats = df.groupby(faixas, observed=True)[target_col].agg(['count', 'mean']).reset_index()
            
            # Formatamos o nome das faixas para ficar bonito no gráfico (ex: (18.0, 25.0])
            labels = stats[feature_name].astype(str).tolist()
        else:
            # Se for categórica ou discreta (ex: UF, Tipo de Produto)
            stats = df.groupby(feature_name)[target_col].agg(['count', 'mean']).reset_index()
            # Se tiver muitas categorias, pegamos as 8 maiores por volume
            stats = stats.sort_values('count', ascending=False).head(8)
            labels = stats[feature_name].astype(str).tolist()

        # 3. Preparação das Séries para o Frontend
        vol = stats['count'].tolist()
        def_rate = (stats['mean'] * 100).round(2).tolist() # Em percentagem

        logger.info(f"Estatísticas geradas com sucesso para a feature: {feature_name}")
        
        return {
            "feature": feature_name,
            "labels": labels,
            "vol": vol,
            "def": def_rate
        }
        
    except Exception as e:
        logger.error(f"Erro ao calcular stats para {feature_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# Rotas Existentes (Métricas e Configuração)
# ---------------------------------------------------------
@app.get("/api/metrics")
async def get_metrics():
    """Lê as métricas reais geradas pelo modelo e envia para o Dashboard."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="Métricas não encontradas.")
    except Exception as e:
        logger.error(f"Erro ao ler métricas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-config")
async def save_config(config_data: PipelineConfig):
    """Recebe as configurações da Web e escreve o ficheiro config.yaml"""
    try:
        yaml_structure = {
            "pipeline_name": config_data.pipeline_name,
            "version": "1.1.0",
            "environment": "development",
            "recipe": {
                "target": config_data.target,
                "features": {
                    "numeric": config_data.numeric_features,
                    "categorical": config_data.categorical_features
                },
                "engineering": {
                    "apply_yeo_johnson_to": config_data.yeo_johnson_features
                }
            },
            "model_training": {
                "algorithm": config_data.algorithm,
                "hyperparameter_tuning": {
                    "auto_tune": config_data.auto_tune,
                    "engine": "optuna"
                }
            }
        }
        
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        
        with open(CONFIG_PATH, 'w') as file:
            yaml.dump(yaml_structure, file, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Configuração guardada em: {CONFIG_PATH}")
        return {"status": "success", "message": "Configuração salva!"}
        
    except Exception as e:
        logger.error(f"Erro ao guardar configuração: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_pipeline_script():
    """Executa o orquestrador em background"""
    script_path = os.path.join(BASE_DIR, "pipeline_orchestrator.py")
    logger.info("Iniciando treinamento...")
    subprocess.run(["python", script_path])

@app.post("/api/run-pipeline")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline_script)
    return {"status": "success", "message": "Pipeline iniciada!"}