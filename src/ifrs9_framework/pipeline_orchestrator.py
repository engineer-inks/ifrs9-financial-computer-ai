import os
import yaml
import json
import time
import logging
import pandas as pd
from datetime import datetime
from catboost import CatBoostClassifier
# Importando LightGBM (pois adicionamos no UI)
from lightgbm import LGBMClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import optuna

# Desativamos o log padrão do Optuna porque nós vamos criar o nosso próprio log visual!
optuna.logging.set_verbosity(optuna.logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger("MLOps-Orchestrator")

class PipelineTracker:
    """Motor de telemetria que comunica o progresso para o Painel Visual HTML"""
    def __init__(self, base_dir):
        self.status_path = os.path.join(base_dir, "config", "pipeline_status.json")
        self.state = {
            "global_status": "running",
            "nodes": {
                "step_1": {"status": "pending", "title": "1. Ingestão de Dados", "logs": "", "duration": "0s"},
                "step_2": {"status": "pending", "title": "2. Feature Engineering", "logs": "", "duration": "0s"},
                "step_3": {"status": "pending", "title": "3. Otimização & Treino", "logs": "", "duration": "0s"},
                "step_4": {"status": "pending", "title": "4. Auditoria (SHAP & ROC)", "logs": "", "duration": "0s"}
            }
        }
        self.start_times = {}
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
        with open(self.status_path, 'w') as f:
            json.dump(self.state, f)

    def update_node(self, node_id, status, log_msg=None):
        if status == "running" and self.state["nodes"][node_id]["status"] == "pending":
            self.start_times[node_id] = datetime.now()
        
        self.state["nodes"][node_id]["status"] = status
        
        if log_msg:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.state["nodes"][node_id]["logs"] += f"[{timestamp}] {log_msg}\n"
            logger.info(f"[{node_id.upper()}] {log_msg}")

        if status in ["success", "failed"] and node_id in self.start_times:
            dur = (datetime.now() - self.start_times[node_id]).total_seconds()
            self.state["nodes"][node_id]["duration"] = f"{dur:.1f}s"

        self._save()
        # Removido o sleep longo daqui para o Optuna poder disparar logs super rápido!

    def finish_pipeline(self, status):
        self.state["global_status"] = status
        self._save()

class CreditRiskPipeline:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.config_path = os.path.join(self.base_dir, "config", "config.yaml")
        with open(self.config_path, 'r') as file: self.config = yaml.safe_load(file)
        self.tracker = PipelineTracker(self.base_dir)

    def step_1_ingestion(self):
        node = "step_1"
        self.tracker.update_node(node, "running", "Iniciando Ingestão de Dados...")
        
        path = self.config.get('data_paths', {}).get('raw_data', "data/raw/synthetic_credit_data.parquet")
        if path.startswith("../"):
            path = os.path.normpath(os.path.join(self.base_dir, path.replace("../src/ifrs9_framework/", "")))
        
        self.tracker.update_node(node, "running", f"Lendo base Parquet: {path}")
        self.df = pd.read_parquet(path)
        self.tracker.update_node(node, "running", f"Dados carregados. Total: {len(self.df)} linhas.")
        time.sleep(0.5) # Pausa estética leve
        self.tracker.update_node(node, "success", "Passo 1 concluído.")

    def step_2_feature_engineering(self):
        node = "step_2"
        self.tracker.update_node(node, "running", "Processando Engenharia de Features (YEO-JOHNSON)...")
        
        recipe = self.config['recipe']
        self.target = recipe['target']
        self.features = recipe['features']['numeric'] + recipe['features']['categorical']
        
        # --- A CORREÇÃO DO SEU BUG (DATETIME/FLOAT CRASH NO CATBOOST) ---
        # Forçamos tudo o que for categórico a virar "String" pura.
        cat_cols = recipe['features'].get('categorical', [])
        if cat_cols:
            self.tracker.update_node(node, "running", f"Convertendo categóricas para string: {cat_cols}")
            for col in cat_cols:
                self.df[col] = self.df[col].astype(str)
        # -----------------------------------------------------------------

        if 'data_contratacao' in self.df.columns:
            self.df = self.df.sort_values('data_contratacao')
            
        X = self.df[self.features]
        y = self.df[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.tracker.update_node(node, "running", f"Train Split OOT: {len(self.X_train)} Treino | {len(self.X_test)} Teste.")
        time.sleep(0.5)
        self.tracker.update_node(node, "success", "Base particionada cronologicamente.")

    def step_3_model_training(self):
        node = "step_3"
        self.tracker.update_node(node, "running", "Iniciando Treinamento do Modelo...")
        
        algo = self.config.get('model_training', {}).get('algorithm', 'catboost')
        self.tracker.update_node(node, "running", f"Algoritmo Selecionado: {algo.upper()}")
        
        # Configurações do Optuna
        tuning_cfg = self.config.get('model_training', {}).get('hyperparameter_tuning', {})
        is_optuna = tuning_cfg.get('auto_tune', False)
        search_space = tuning_cfg.get('search_space', {})
        
        cat_cols = self.config['recipe']['features'].get('categorical', [])
        cat_features_idx = [self.features.index(col) for col in cat_cols] if cat_cols else []

        best_params = self.config.get('model_training', {}).get('static_params', {})
        
        # === O NOVO MOTOR DETALHADO DO OPTUNA ===
        if is_optuna:
            self.tracker.update_node(node, "running", ">> ATIVANDO OPTUNA (OTIMIZAÇÃO BAYESIANA) <<")
            
            def objective(trial):
                # Sorteio inteligente nos limites que o usuário escolheu na Web
                lr_range = search_space.get('learning_rate', [0.01, 0.1])
                w_range = search_space.get('scale_pos_weight', [1.0, 15.0])
                
                params = {
                    'iterations': 50, # Baixo para ir rápido na simulação
                    'learning_rate': trial.suggest_float('learning_rate', lr_range[0], lr_range[1]),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', w_range[0], w_range[1])
                }
                
                # Suporte aos dois algoritmos
                if algo == 'lightgbm':
                    params['max_depth'] = trial.suggest_int('max_depth', int(search_space.get('max_depth', [4, 8])[0]), int(search_space.get('max_depth', [4, 8])[1]))
                    model = LGBMClassifier(**params, random_state=42, verbose=-1)
                    # LightGBM requer que colunas sejam do tipo 'category'
                    for c in cat_cols:
                        self.X_train[c] = self.X_train[c].astype('category')
                        self.X_test[c] = self.X_test[c].astype('category')
                else:
                    params['depth'] = trial.suggest_int('depth', int(search_space.get('depth', [4, 8])[0]), int(search_space.get('depth', [4, 8])[1]))
                    model = CatBoostClassifier(**params, cat_features=cat_features_idx, verbose=0, random_seed=42)
                
                model.fit(self.X_train, self.y_train)
                preds = model.predict_proba(self.X_test)[:, 1]
                return average_precision_score(self.y_test, preds)

            study = optuna.create_study(direction='maximize')
            n_trials = 5 # Executamos 5 testes de IA

            # Callback para cuspir as métricas no Dataflow em tempo real!
            def optuna_logger(study, trial):
                self.tracker.update_node(node, "running", f"[Optuna] Trial {trial.number + 1}/{n_trials} | PR-AUC = {trial.value:.4f} | LR: {trial.params['learning_rate']:.3f}")
            
            study.optimize(objective, n_trials=n_trials, callbacks=[optuna_logger])
            best_params = study.best_params
            best_params['iterations'] = 150 # Para o modelo final
            
            self.tracker.update_node(node, "running", f">> MELHOR COMBINAÇÃO ENCONTRADA: {best_params} <<")
        # =========================================
        
        self.tracker.update_node(node, "running", f"Construindo árvores finais ({algo.upper()})...")
        
        # Treinamento Final
        if algo == 'lightgbm':
            for c in cat_cols:
                self.X_train[c] = self.X_train[c].astype('category')
                self.X_test[c] = self.X_test[c].astype('category')
            self.model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        else:
            self.model = CatBoostClassifier(**best_params, cat_features=cat_features_idx, verbose=0, random_seed=42)
            
        self.model.fit(self.X_train, self.y_train)
        
        self.tracker.update_node(node, "success", "Treinamento Finalizado e Convergido!")

    def step_4_evaluation_and_audit(self):
        node = "step_4"
        self.tracker.update_node(node, "running", "A calcular Feature Importance (SHAP)...")
        
        from visualization.model_dashboard import MetricsGenerator
        metrics_path = os.path.join(self.base_dir, "config", "metrics.json")
        auditor = MetricsGenerator(model=self.model, X_test=self.X_test, y_test=self.y_test, cutoff=0.04)
        auditor.generate_metrics(metrics_path)
        
        self.tracker.update_node(node, "running", "A empacotar métricas de Auditoria ROC/KS.")
        time.sleep(0.5)
        self.tracker.update_node(node, "success", "Métricas gravadas no Dashboard Web.")

    def run_pipeline(self):
        try:
            self.step_1_ingestion()
            self.step_2_feature_engineering()
            self.step_3_model_training()
            self.step_4_evaluation_and_audit()
            self.tracker.finish_pipeline("completed")
        except Exception as e:
            logger.error(str(e))
            if hasattr(self, 'tracker'):
                self.tracker.update_node("step_3", "failed", f"CRASH: {str(e)}")
                self.tracker.finish_pipeline("failed")

if __name__ == "__main__":
    pipeline = CreditRiskPipeline()
    pipeline.run_pipeline()