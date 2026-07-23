import os
import json
import yaml
import logging
import multiprocessing
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Extra
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MLOps-API")

app = FastAPI(title="IFRS 9 MLOps Autonomous Pipeline API", version="2.9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")

if os.path.exists(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

@app.get("/")
async def root():
    return RedirectResponse(url="/ui/config_ui.html")

class PipelineConfig(BaseModel):
    pipeline_name: Optional[str] = "ifrs9_credit_origination"
    target_column: Optional[str] = "default_flag"
    group_column: Optional[str] = "codigo_contrato"
    numeric_features: Optional[List[str]] = []
    categorical_features: Optional[List[str]] = []
    yeo_johnson_features: Optional[List[str]] = []
    algorithm: Optional[str] = "catboost"
    loss_function: Optional[str] = "logloss"
    auto_tune: Optional[bool] = False
    hyperparameters: Optional[Dict[str, Any]] = {}

    class Config:
        extra = Extra.allow

class DatasetLoadRequest(BaseModel):
    conn_type: str = "local"
    file_path: str

current_dataset = None
running_pipeline_process: Optional[multiprocessing.Process] = None

def reset_pipeline_status():
    status_path = os.path.join(BASE_DIR, "config", "pipeline_status.json")
    default_status = {
        "global_status": "pending",
        "nodes": {
            "step_1": {"status": "pending", "title": "1. Ingestão de Dados", "logs": "", "duration": "0s"},
            "step_2": {"status": "pending", "title": "2. Feature Engineering", "logs": "", "duration": "0s"},
            "step_3": {"status": "pending", "title": "3. Otimização & Treino", "logs": "", "duration": "0s"},
            "step_4": {"status": "pending", "title": "4. Calibração OOF & Deploy", "logs": "", "duration": "0s"}
        }
    }
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(default_status, f, ensure_ascii=False)
    logger.info("Cache de execução limpo. Estado reiniciado para novo ciclo.")

@app.post("/api/load-dataset")
async def load_dataset(req: DatasetLoadRequest):
    global current_dataset
    try:
        path = req.file_path
        if path.startswith("../"):
            path = os.path.normpath(os.path.join(BASE_DIR, path.replace("../src/ifrs9_framework/", "")))
        if not os.path.exists(path):
            path = req.file_path
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Ficheiro não encontrado em: {path}")
        
        if path.endswith('.parquet'):
            current_dataset = pd.read_parquet(path)
        elif path.endswith('.csv'):
            current_dataset = pd.read_csv(path)
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado.")
        
        reset_pipeline_status()
        
        logger.info(f"Dataset carregado com sucesso. Total de linhas: {len(current_dataset)}")
        preview_df = current_dataset.head(15).where(pd.notnull(current_dataset.head(15)), None)
        return {
            "total_rows": len(current_dataset),
            "columns": [str(c) for c in current_dataset.columns],
            "preview": preview_df.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset-schema")
async def get_dataset_schema():
    global current_dataset
    if current_dataset is None:
        default_path = os.path.join(BASE_DIR, "data", "raw", "synthetic_credit_data.parquet")
        if os.path.exists(default_path):
            current_dataset = pd.read_parquet(default_path)
        else:
            raise HTTPException(status_code=400, detail="Nenhum dataset carregado.")
    schema = []
    for col in current_dataset.columns:
        col_str = str(col)
        if col_str in ['default_flag', 'IDC_DFT_POS_RFC']:
            role = 'target'
        elif pd.api.types.is_numeric_dtype(current_dataset[col]):
            role = 'numeric'
        else:
            role = 'categorical'
        schema.append({"name": col_str, "role": role})
    return schema

@app.get("/api/feature-stats/{feature_name}")
async def get_feature_stats(feature_name: str):
    global current_dataset
    if current_dataset is None or feature_name not in current_dataset.columns:
        raise HTTPException(status_code=404, detail="Feature ou dataset não encontrado.")
    target_col = 'default_flag' if 'default_flag' in current_dataset.columns else current_dataset.columns[0]
    try:
        s = current_dataset[feature_name]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            binned = pd.qcut(s, q=10, duplicates='drop')
            agg = current_dataset.groupby(binned, observed=False).agg(
                vol=(feature_name, 'count'), def_rate=(target_col, 'mean')
            ).reset_index()
            labels = [str(interval) for interval in agg[feature_name]]
            vol = agg['vol'].tolist()
            def_rate = (agg['def_rate'] * 100).tolist()
        else:
            agg = current_dataset.groupby(feature_name, observed=False).agg(
                vol=(feature_name, 'count'), def_rate=(target_col, 'mean')
            ).reset_index().head(15)
            labels = [str(val) for val in agg[feature_name]]
            vol = agg['vol'].tolist()
            def_rate = (agg['def_rate'] * 100).tolist()
        return {"labels": labels, "vol": vol, "def": def_rate}
    except Exception:
        return {"labels": ["Val 1", "Val 2"], "vol": [100, 200], "def": [5.0, 10.0]}

@app.post("/api/save-config")
async def save_config(config: PipelineConfig):
    try:
        config_dir = os.path.join(BASE_DIR, "config")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "config.yaml")
        config_dict = config.dict()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Configuração YAML guardada em: {config_path}")
        return {"status": "success", "message": "Configuração salva com sucesso!", "config": config_dict}
    except Exception as e:
        logger.error(f"Erro ao salvar configuração: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/load-config")
async def load_config():
    try:
        config_path = os.path.join(BASE_DIR, "config", "config.yaml")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="Configuração não encontrada.")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return config_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def target_run_pipeline():
    try:
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        from pipeline_orchestrator import CreditRiskPipeline
        pipeline = CreditRiskPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Erro crítico no processo da pipeline: {str(e)}", exc_info=True)

@app.post("/api/run-pipeline")
async def run_pipeline():
    global running_pipeline_process
    try:
        if running_pipeline_process is not None and running_pipeline_process.is_alive():
            running_pipeline_process.terminate()
            running_pipeline_process.join()

        reset_pipeline_status()

        running_pipeline_process = multiprocessing.Process(target=target_run_pipeline)
        running_pipeline_process.start()
        return {"status": "success", "message": "Pipeline acionada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline-status")
async def get_pipeline_status():
    global running_pipeline_process
    status_path = os.path.join(BASE_DIR, "config", "pipeline_status.json")
    
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    
                    # OOM Watchdog: O processo morreu misteriosamente?
                    if data.get("global_status") == "running" and running_pipeline_process is not None:
                        if not running_pipeline_process.is_alive():
                            logger.error("WATCHDOG: Processo Python morreu subitamente! (Possível OOM Kill)")
                            data["global_status"] = "failed"
                            
                            # Encontra o nó rodando e marca falha crítica
                            for node_id, node_info in data.get("nodes", {}).items():
                                if node_info["status"] == "running":
                                    node_info["status"] = "failed"
                                    node_info["logs"] += "\n\n[FATAL ERROR] O processo morreu subitamente (Possível OOM Kill / Falta de RAM do Docker)."
                                    
                            with open(status_path, "w", encoding="utf-8") as fw:
                                json.dump(data, fw, ensure_ascii=False)
                                
                    return data
        except Exception:
            pass
            
    return {"global_status": "pending"}

@app.post("/api/cancel-pipeline")
async def cancel_pipeline():
    global running_pipeline_process
    try:
        if running_pipeline_process is not None and running_pipeline_process.is_alive():
            running_pipeline_process.terminate()
            running_pipeline_process.join()
            logger.info("Processo terminado.")

        status_path = os.path.join(BASE_DIR, "config", "pipeline_status.json")
        if os.path.exists(status_path):
            with open(status_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read().strip() or "{}")
        else:
            data = {}
            
        data["global_status"] = "cancelled"
        for node_id in data.get("nodes", {}):
            if data["nodes"][node_id]["status"] == "running":
                data["nodes"][node_id]["status"] = "failed"
                data["nodes"][node_id]["logs"] += "\n[ABORTADO PELO UTILIZADOR]"

        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return {"status": "success", "message": "Pipeline abortada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))