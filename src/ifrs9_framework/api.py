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

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLOps-API")

# ==========================================
# INICIALIZAÇÃO DA APP (O Uvicorn procura por isto!)
# ==========================================
app = FastAPI(title="IFRS9 MLOps API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Estruturas de Dados (Payloads) ---
class PipelineConfig(BaseModel):
    pipeline_name: str
    target: str
    numeric_features: list
    categorical_features: list
    yeo_johnson_features: list
    algorithm: str
    auto_tune: bool
    hyperparameters: dict = {}

class ConnectionInfo(BaseModel):
    conn_type: str
    file_path: str = None

# --- Caminhos do Sistema ---
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
METRICS_PATH = os.path.join(BASE_DIR, "config", "metrics.json")
STATUS_PATH = os.path.join(BASE_DIR, "config", "pipeline_status.json")

# Variável global para rastrear o processo de treinamento em background (Dataflow)
current_pipeline_process = None

def get_dynamic_data_path():
    """Lê o config.yaml e retorna o caminho real da base de dados."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as file:
                config = yaml.safe_load(file)
                if 'data_paths' in config and 'raw_data' in config['data_paths']:
                    path = config['data_paths']['raw_data']
                    if path.startswith("../"):
                        return os.path.normpath(os.path.join(BASE_DIR, path.replace("../src/ifrs9_framework/", "")))
                    return path
    except Exception as e:
        logger.warning(f"Erro ao ler caminho: {e}")
    return os.path.join(BASE_DIR, "data", "raw", "synthetic_credit_data.parquet")

# ==========================================
# ROTAS DA API
# ==========================================

@app.post("/api/load-dataset")
async def load_dataset(connection_info: ConnectionInfo):
    """Testa conexão com a base e retorna as 15 primeiras linhas."""
    try:
        if connection_info.conn_type == "local":
            path = connection_info.file_path
            path_resolved = os.path.normpath(os.path.join(BASE_DIR, path.replace("../src/ifrs9_framework/", ""))) if path.startswith("../") else path

            if not os.path.exists(path_resolved):
                raise HTTPException(status_code=404, detail=f"Ficheiro não encontrado: {path_resolved}")
            
            yaml_config = {}
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as file: yaml_config = yaml.safe_load(file) or {}
                
            if 'data_paths' not in yaml_config: yaml_config['data_paths'] = {}
            yaml_config['data_paths']['raw_data'] = connection_info.file_path
            
            with open(CONFIG_PATH, 'w') as file:
                yaml.dump(yaml_config, file, default_flow_style=False, sort_keys=False)
            
            df = pd.read_parquet(path_resolved)
            preview_data = df.head(15).fillna("").to_dict(orient='records')
            return {"status": "connected", "preview": preview_data, "columns": list(df.columns), "total_rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset-schema")
async def get_dataset_schema():
    """Devolve a lista de colunas para construir as caixas de Drag and Drop."""
    try:
        data_path = get_dynamic_data_path()
        if not os.path.exists(data_path): raise HTTPException(status_code=404, detail="Parquet não encontrado.")
        df = pd.read_parquet(data_path)
        schema = [{"name": col, "role": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"} for col in df.columns if col != "default_flag"]
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feature-stats/{feature_name}")
async def get_feature_stats(feature_name: str):
    """Devolve os dados para construir o gráfico da variável."""
    try:
        data_path = get_dynamic_data_path()
        df = pd.read_parquet(data_path, columns=[feature_name, "default_flag"])
        
        if pd.api.types.is_numeric_dtype(df[feature_name]) and df[feature_name].nunique() > 10:
            faixas = pd.qcut(df[feature_name], q=5, duplicates='drop')
            stats = df.groupby(faixas, observed=True)["default_flag"].agg(['count', 'mean']).reset_index()
            labels = stats[feature_name].astype(str).tolist()
        else:
            stats = df.groupby(feature_name)["default_flag"].agg(['count', 'mean']).reset_index().sort_values('count', ascending=False).head(8)
            labels = stats[feature_name].astype(str).tolist()

        return {"feature": feature_name, "labels": labels, "vol": stats['count'].tolist(), "def": (stats['mean'] * 100).round(2).tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-config")
async def save_config(config_data: PipelineConfig):
    """Guarda a Receita do Modelo e os Hiperparâmetros no ficheiro YAML."""
    try:
        search_space = config_data.hyperparameters if config_data.auto_tune else {}
        static_params = config_data.hyperparameters if not config_data.auto_tune else {}
        if 'iterations' not in static_params: static_params['iterations'] = 300

        yaml_config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as file: yaml_config = yaml.safe_load(file) or {}

        yaml_structure = {
            "pipeline_name": config_data.pipeline_name,
            "version": "1.1.0",
            "environment": "development",
            "data_paths": yaml_config.get('data_paths', {"raw_data": "../src/ifrs9_framework/data/raw/synthetic_credit_data.parquet"}),
            "recipe": {
                "target": config_data.target,
                "features": {"numeric": config_data.numeric_features, "categorical": config_data.categorical_features},
                "engineering": {"apply_yeo_johnson_to": config_data.yeo_johnson_features}
            },
            "model_training": {
                "algorithm": config_data.algorithm,
                "hyperparameter_tuning": {"auto_tune": config_data.auto_tune, "engine": "optuna", "search_space": search_space},
                "static_params": static_params   
            }
        }
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as file: yaml.dump(yaml_structure, file, default_flow_style=False, sort_keys=False)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ROTAS DA FASE 3 (DATAFLOW & CANCELAMENTO)
# ==========================================

@app.get("/api/pipeline-status")
async def get_pipeline_status():
    """Usado pelo HTML para saber em que passo o Python está a trabalhar."""
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH, 'r') as f: return json.load(f)
        return {"global_status": "idle", "nodes": {}}
    except Exception as e:
        return {"global_status": "error"}

@app.post("/api/cancel-pipeline")
async def cancel_pipeline():
    """Mata o processo de treinamento em background."""
    global current_pipeline_process
    
    if current_pipeline_process is not None and current_pipeline_process.poll() is None:
        logger.info("Encerrando processo de treinamento a pedido do usuário...")
        current_pipeline_process.terminate()
        current_pipeline_process = None
        
        try:
            if os.path.exists(STATUS_PATH):
                with open(STATUS_PATH, 'r') as f:
                    state = json.load(f)
                
                state["global_status"] = "cancelled"
                for node_id, info in state.get("nodes", {}).items():
                    if info.get("status") == "running":
                        info["status"] = "failed"
                        info["logs"] = info.get("logs", "") + "\n[SISTEMA] 🛑 Treinamento abortado pelo utilizador."
                
                with open(STATUS_PATH, 'w') as f:
                    json.dump(state, f)
        except Exception as e:
            logger.error(f"Erro ao atualizar status: {e}")
            
        return {"status": "success", "message": "Treinamento Cancelado."}
    
    return {"status": "ignored", "message": "Nenhum treinamento em execução."}

def run_pipeline_script():
    """Função executada na Thread de Background (Subprocess)"""
    global current_pipeline_process
    if os.path.exists(STATUS_PATH): os.remove(STATUS_PATH)
    
    script_path = os.path.join(BASE_DIR, "pipeline_orchestrator.py")
    current_pipeline_process = subprocess.Popen(["python", script_path])
    current_pipeline_process.wait() # Aguarda terminar
    current_pipeline_process = None

@app.post("/api/run-pipeline")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """Inicia o Treinamento"""
    global current_pipeline_process
    
    # Previne o cientista de clicar no botão "treinar" 2 vezes e encravar o servidor
    if current_pipeline_process is not None and current_pipeline_process.poll() is None:
        raise HTTPException(status_code=400, detail="Um treinamento já está em execução!")
        
    background_tasks.add_task(run_pipeline_script)
    return {"status": "success"}