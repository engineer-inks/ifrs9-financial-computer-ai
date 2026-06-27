import os
import yaml
import logging
import pandas as pd
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import optuna

# Importando o nosso motor analítico que você acabou de criar
from visualization.model_dashboard import MetricsGenerator

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suprimindo logs excessivos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class CreditRiskPipeline:
    def __init__(self):
        logging.info("=== Inicializando Pipeline MLOps IFRS9 ===")
        self.base_dir = os.path.dirname(__file__)
        self.config_path = os.path.join(self.base_dir, "config", "config.yaml")
        
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.df = None
        self.model = None
        logging.info(f"Receita carregada: {self.config['pipeline_name']} (v{self.config['version']})")

    def step_1_ingestion(self):
        logging.info(">>> PASSO 1: Ingestão de Dados")
        data_path = os.path.join(self.base_dir, "data", "raw", "synthetic_credit_data.parquet")
        logging.info(f"Lendo dados de {data_path}...")
        self.df = pd.read_parquet(data_path)
        logging.info(f"Dados carregados! Total de registos: {len(self.df)}")

    def step_2_feature_engineering(self):
        logging.info(">>> PASSO 2: Preparação da Base (OOT Split)")
        recipe = self.config['recipe']
        self.target = recipe['target']
        self.features = recipe['features']['numeric'] + recipe['features']['categorical']
        self.cat_features = recipe['features']['categorical']
        
        if 'data_contratacao' in self.df.columns:
            self.df = self.df.sort_values('data_contratacao')
            
        X = self.df[self.features]
        y = self.df[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        logging.info(f"Treino: {len(self.X_train)} contratos | Teste OOT: {len(self.X_test)} contratos.")

    def optimize_hyperparameters(self):
        """Usa Optuna para encontrar a melhor combinação para maximizar a precisão."""
        logging.info("Iniciando Otimização Bayesiana com Optuna (Pode demorar)...")
        
        cat_features_idx = [self.features.index(col) for col in self.cat_features]
        
        def objective(trial):
            params = {
                'iterations': 100, # Mantemos baixo para a otimização ser rápida
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 15.0) 
            }
            
            model = CatBoostClassifier(**params, cat_features=cat_features_idx, random_seed=42, verbose=0)
            model.fit(self.X_train, self.y_train)
            preds_proba = model.predict_proba(self.X_test)[:, 1]
            return average_precision_score(self.y_test, preds_proba)

        study = optuna.create_study(direction='maximize')
        # Reduzimos para 5 trials apenas para demonstração local. Em prod seria 50+
        study.optimize(objective, n_trials=5)
        
        logging.info(f"Melhores parâmetros encontrados: {study.best_params}")
        return study.best_params

    def step_3_model_training(self):
        logging.info(">>> PASSO 3: Treinamento do Modelo (CatBoost)")
        
        cat_features_idx = [self.features.index(col) for col in self.cat_features]
        
        # Lê a configuração para ver se o Cientista ativou o Optuna na Tela HTML
        tuning_config = self.config.get('model_training', {}).get('hyperparameter_tuning', {})
        
        if tuning_config.get('auto_tune', False):
            logging.info("[AUTO-TUNE ATIVADO] Buscando hiperparâmetros dinâmicos...")
            best_params = self.optimize_hyperparameters()
            best_params['iterations'] = 300 # Aumentamos as árvores para o treino final
        else:
            logging.info("[AUTO-TUNE DESATIVADO] Usando parâmetros estáticos do YAML.")
            best_params = {
                'iterations': 300,
                'learning_rate': 0.05,
                'depth': 6,
                'scale_pos_weight': 10
            }
        
        self.model = CatBoostClassifier(
            **best_params,
            cat_features=cat_features_idx,
            eval_metric='AUC',
            random_seed=42,
            verbose=50
        )
        
        logging.info("Iniciando fit() do modelo final campeão...")
        self.model.fit(self.X_train, self.y_train)
        logging.info("Treinamento concluído com sucesso!")

    def step_4_evaluation_and_audit(self):
        logging.info(">>> PASSO 4: Auditoria e Geração de Métricas Web")
        metrics_path = os.path.join(self.base_dir, "config", "metrics.json")
        auditor = MetricsGenerator(
            model=self.model, X_test=self.X_test, y_test=self.y_test, cutoff=0.04
        )
        auditor.generate_metrics(metrics_path)
        logging.info("Painel de Auditoria alimentado com novos dados!")

    def run_pipeline(self):
        start_time = datetime.now()
        logging.info("=== START: PIPELINE DE ORIGINAÇÃO ===")
        try:
            self.step_1_ingestion()
            self.step_2_feature_engineering()
            self.step_3_model_training()
            self.step_4_evaluation_and_audit()
            duration = datetime.now() - start_time
            logging.info(f"=== SUCESSO! TEMPO TOTAL: {duration} ===")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL: {str(e)} !!!")

if __name__ == "__main__":
    pipeline = CreditRiskPipeline()
    pipeline.run_pipeline()