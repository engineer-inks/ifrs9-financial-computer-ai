import os
import sys
import yaml
import json
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from core_utils import configurar_logging, apply_custom_plot_style
    from features_pipeline import load_and_filter_data, features_engineer, build_preprocessor, get_focal_loss_obj
    from advanced_metrics_and_split import div_train_test_split, gerar_metricas_classificacao, salvar_metricas_csv
    from model_training_engine import preparar_matrizes, search_hyperparameters_and_training_model
    from hyper_tuner_advanced import otimizar_hiperparametros_grid_search
    from calibration_engine import recalibrar_probabilidade_shift
    from oof_calibration_engine import optimization_threshold, gerar_grafico_resumo_kpis_completo, calcular_metricas_performance
    from reporting_engine import plotar_feature_importance, gerar_analise_temporal
    from evaluation_engine import salvar_pipeline_completo, executar_teste_aderencia_buckets
    from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score, precision_score
    from scipy.stats import ks_2samp
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    ERRO_IMPORT = str(e)

logger = logging.getLogger("MLOps-Orchestrator")
logger.setLevel(logging.INFO)

class PipelineTracker:
    def __init__(self, base_dir):
        self.status_path = os.path.join(base_dir, "config", "pipeline_status.json")
        self.state = {
            "global_status": "running",
            "nodes": {
                "step_1": {"status": "pending", "title": "1. Ingestão & Filtros", "logs": "", "duration": "0s"},
                "step_2": {"status": "pending", "title": "2. Feature Engineering", "logs": "", "duration": "0s"},
                "step_3": {"status": "pending", "title": "3. CatBoost & Tuning", "logs": "", "duration": "0s"},
                "step_4": {"status": "pending", "title": "4. Calibração OOF & Deploy", "logs": "", "duration": "0s"}
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

    def finish_pipeline(self, status):
        self.state["global_status"] = status
        self._save()

class CreditRiskPipeline:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_path = os.path.join(self.base_dir, "config", "config.yaml")
        
        self.config = {
            'target_column': 'default_flag',
            'group_column': 'COD_OPR_ATV',
            'test_size': 0.25,
            'random_state': 42,
            'GRAPHICS_DIR': os.path.join(self.base_dir, "outputs", "graphics"),
            'metrics': os.path.join(self.base_dir, "outputs"),
            'models_dir': os.path.join(self.base_dir, "models")
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file: 
                loaded_config = yaml.safe_load(file)
                if loaded_config:
                    self.config.update(loaded_config)
                    
        if not self.config.get('GRAPHICS_DIR'):
            self.config['GRAPHICS_DIR'] = os.path.join(self.base_dir, "outputs", "graphics")
        if not self.config.get('metrics'):
            self.config['metrics'] = os.path.join(self.base_dir, "outputs")
        if not self.config.get('models_dir'):
            self.config['models_dir'] = os.path.join(self.base_dir, "models")
            
        self.tracker = PipelineTracker(self.base_dir)
        if MODULES_LOADED:
            apply_custom_plot_style()

    def step_1_ingestion(self):
        node = "step_1"
        self.tracker.update_node(node, "running", "Iniciando Ingestão de Dados...")
        
        if not MODULES_LOADED:
            raise ImportError(f"Falha ao carregar motores modulares: {ERRO_IMPORT}")

        self.df = load_and_filter_data(self.config, logger)
        
        group_col = self.config.get('group_column', 'COD_OPR_ATV')
        if group_col not in self.df.columns:
            self.tracker.update_node(node, "running", "⚠️ Coluna de grupo não encontrada. Criando proxy...")
            self.df[group_col] = np.arange(len(self.df))
            
        self.tracker.update_node(node, "success", f"Ingestão concluída. Total: {len(self.df):,} linhas.")

    def step_2_feature_engineering(self):
        node = "step_2"
        self.tracker.update_node(node, "running", "Processando Engenharia de Features...")
        
        self.df_eng = features_engineer(self.df, self.config, logger)
        
        self.tracker.update_node(node, "running", "Aplicando Group Split (Treino vs Teste)...")
        self.df_train_val, self.df_test = div_train_test_split(self.df_eng, self.config)
        
        self.tracker.update_node(node, "running", "Limpando constantes e aplicando pré-processador...")
        self.data_dict = preparar_matrizes(self.df_train_val, self.df_test, self.config, build_preprocessor)
        
        self.tracker.update_node(node, "success", "Engenharia de features e split concluídos.")

    def step_3_model_training(self):
        node = "step_3"
        self.tracker.update_node(node, "running", "Iniciando Otimização de Hiperparâmetros...")
        
        tuning_cfg = self.config.get('model_training', {}).get('hyperparameter_tuning', {})
        is_optuna = tuning_cfg.get('auto_tune', False)
        
        self.model_final, self.melhores_params = search_hyperparameters_and_training_model(
            config=self.config,
            data_dict=self.data_dict,
            tuner_func=otimizar_hiperparametros_grid_search,
            get_focal_loss_obj_func=get_focal_loss_obj,
            recalibrar_func=recalibrar_probabilidade_shift,
            sample_base=True,
            search_method='grid' if is_optuna else 'focal',
            metodo='logloss'
        )
        
        self.tracker.update_node(node, "success", "Modelo otimizado e treinado na base completa.")

    def step_4_evaluation_and_audit(self):
        node = "step_4"
        self.tracker.update_node(node, "running", "Iniciando Calibração Avançada OOF e Relatórios IFRS 9...")
        
        calibrator, resultado_otimo, f1_h, y_prob_test_final, y_pred_test = optimization_threshold(
            config=self.config,
            model_final=self.model_final,
            X_train_val_processed=self.data_dict['X_train_val_processed'],
            y_train_val=self.data_dict['y_train_val'],
            groups_train_val=self.data_dict['groups_train'],
            melhores_params=self.melhores_params,
            X_test_processed=self.data_dict['X_test_processed'],
            y_test=self.data_dict['y_test'],
            cat_indices=self.data_dict['cat_indices'],
            preprocessor=self.data_dict['preprocessor'],
            sample=True,
            method='isotonic'
        )
        
        t_otimo = resultado_otimo['threshold']
        self.tracker.update_node(node, "running", f"Threshold Otimizado: {t_otimo:.2%} | Ratio: {resultado_otimo['peso']:.2f}x")
        
        y_test = self.data_dict['y_test']
        auc = roc_auc_score(y_test, y_prob_test_final)
        ks_stat, _ = ks_2samp(y_prob_test_final[y_test == 0], y_prob_test_final[y_test == 1])
        acc, mcc = calcular_metricas_performance(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec = recall_score(y_test, y_pred_test, zero_division=0)
        brier = brier_score_loss(y_test, y_prob_test_final)
        
        out_dir = self.config.get('GRAPHICS_DIR')
        if not out_dir:
            out_dir = os.path.join(self.base_dir, "outputs", "graphics")
            
        gerar_grafico_resumo_kpis_completo(auc, ks_stat, acc, mcc, f1_h, prec, rec, out_dir)
        matriz_res = gerar_metricas_classificacao(y_test, y_prob_test_final, out_dir, t_otimo, method='Isotonic_OOF')
        executar_teste_aderencia_buckets(y_test, y_prob_test_final, n_bins=10)
        plotar_feature_importance(self.model_final, self.data_dict['feature_names'], self.data_dict['X_test_processed'], self.config, out_dir)
        
        if 'DTA_INI_OPR' in self.df_test.columns:
            gerar_analise_temporal(self.df_test, y_test, y_prob_test_final, out_dir, 'DTA_INI_OPR', 'Safra (Início Opr)', 'vintage_safra')

        metrics = {
            'AUC': auc, 'KS': ks_stat, 'F1': f1_h, 'Precision': prec, 'Recall': rec, 'Brier': brier,
            'Metodo_Calibracao': 'isotonic', 'Data': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        metrics.update(matriz_res)
        salvar_metricas_csv(self.config, metrics)
        
        salvar_pipeline_completo(
            config=self.config,
            preprocessor=self.data_dict['preprocessor'],
            model=self.model_final,
            threshold=t_otimo,
            feature_names=self.data_dict['feature_names'],
            cat_indices=self.data_dict['cat_indices'],
            scale_pos_weight=self.melhores_params.get('w_train_calculado', 1.0),
            isotonic_model=calibrator,
            nome_arquivo="pipeline_modelagem_rede_v1.joblib"
        )
        
        self.tracker.update_node(node, "success", "Auditoria IFRS 9 e Empacotamento Concluídos!")

    def run_pipeline(self):
        try:
            self.step_1_ingestion()
            self.step_2_feature_engineering()
            self.step_3_model_training()
            self.step_4_evaluation_and_audit()
            
            metrics_dir = self.config.get('metrics', os.path.join(self.base_dir, "config"))
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump({"status": "Success", "model": "CatBoost_IFRS9_OOF"}, f)
                
            self.tracker.finish_pipeline("completed")
            logger.info("🎉 CICLO COMPLETO FINALIZADO! ARTEFATOS PRONTOS.")
            
        except Exception as e:
            logger.error(f"CRASH NA PIPELINE: {str(e)}", exc_info=True)
            self.tracker.update_node("step_4", "failed", f"CRASH: {str(e)}")
            self.tracker.finish_pipeline("failed")

if __name__ == "__main__":
    pipeline = CreditRiskPipeline()
    pipeline.run_pipeline()