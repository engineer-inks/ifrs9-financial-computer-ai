from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import subprocess
import os
import logging

# Configuração simples de log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLOps-API")

app = FastAPI(title="IFRS9 MLOps API")

# Configurando CORS para permitir que o HTML (mesmo aberto direto no navegador) consiga falar com a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em produção, limitamos isso para o domínio correto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definindo a estrutura que esperamos receber do Frontend (HTML)
class PipelineConfig(BaseModel):
    pipeline_name: str
    target: str
    numeric_features: list
    categorical_features: list
    yeo_johnson_features: list
    algorithm: str
    auto_tune: bool

# Caminho onde vamos salvar o arquivo YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../ifrs9_framework/config/config.yaml")

@app.post("/api/save-config")
async def save_config(config_data: PipelineConfig):
    """Recebe as configurações da interface Web e escreve o arquivo config.yaml no disco."""
    try:
        # Montando a estrutura exata que o nosso pipeline_orchestrator espera
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
        
        # Garantindo que a pasta config existe
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        
        # Salvando no disco
        with open(CONFIG_PATH, 'w') as file:
            yaml.dump(yaml_structure, file, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Arquivo de configuração salvo com sucesso em: {CONFIG_PATH}")
        return {"status": "success", "message": "Configuração salva com sucesso!"}
        
    except Exception as e:
        logger.error(f"Erro ao salvar configuração: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_pipeline_script():
    """Função que roda o orquestrador em background (segundo plano)"""
    script_path = os.path.join(os.path.dirname(__file__), "../pipeline_orchestrator.py")
    logger.info("Iniciando treinamento do modelo em background...")
    # Executa o script Python do nosso orquestrador
    subprocess.run(["python", script_path])

@app.post("/api/run-pipeline")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """Endpoint para iniciar o treinamento do modelo pelo clique do botão."""
    background_tasks.add_task(run_pipeline_script)
    return {"status": "success", "message": "Pipeline iniciada em background! Verifique os logs."}