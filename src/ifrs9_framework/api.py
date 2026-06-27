from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import subprocess
import os
import json
import logging

# Configuração simples de log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLOps-API")

app = FastAPI(title="IFRS9 MLOps API")

# Configurando CORS
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

# Caminhos baseados na pasta src/ifrs9_framework
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/config.yaml")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "config/metrics.json")

@app.get("/api/metrics")
async def get_metrics():
    """Lê as métricas reais geradas pelo modelo em Python e as envia para o Dashboard HTML."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="Métricas ainda não geradas. Inicie um treinamento primeiro.")
    except Exception as e:
        logger.error(f"Erro ao ler métricas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-config")
async def save_config(config_data: PipelineConfig):
    """Recebe as configurações da interface Web e escreve o arquivo config.yaml no disco."""
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
            
        logger.info(f"Arquivo de configuração salvo com sucesso em: {CONFIG_PATH}")
        return {"status": "success", "message": "Configuração salva com sucesso!"}
        
    except Exception as e:
        logger.error(f"Erro ao salvar configuração: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_pipeline_script():
    """Função que roda o orquestrador em background"""
    script_path = os.path.join(os.path.dirname(__file__), "pipeline_orchestrator.py")
    logger.info("Iniciando treinamento do modelo em background...")
    subprocess.run(["python", script_path])

@app.post("/api/run-pipeline")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """Endpoint para iniciar o treinamento do modelo pelo clique do botão."""
    background_tasks.add_task(run_pipeline_script)
    return {"status": "success", "message": "Pipeline iniciada em background! Verifique os logs."}