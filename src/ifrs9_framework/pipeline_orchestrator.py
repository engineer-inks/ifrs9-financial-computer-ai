import json
import logging
import pandas as pd
from datetime import datetime
import os

# Importações dos nossos módulos encapsulados (simulando a estrutura de pastas)
# from src.data_ingestion import read_parquet
# from src.feature_engineering import apply_yeo_johnson, detect_anomalies
# from src.model_training import train_model_with_optuna, train_static_model
# from src.evaluation import evaluate_and_calibrate
from src.features.feature_builder import FeatureEngineer
from src.visualization.model_dashboard import ModelVisualizer

# Configuração de Logs (O que vai alimentar a nossa interface HTML no futuro)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("pipeline_execution.log"),
        logging.StreamHandler()
    ]
)

class CreditRiskPipeline:
    def __init__(self, config_path: str):
        logging.info("Inicializando Pipeline de Originação de Crédito...")
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        self.df = None
        self.model = None
        logging.info(f"Configuração carregada: Versão {self.config['version']} | Ambiente: {self.config['environment']}")
        
        # Verificação de Hardware Dinâmica (Suporte para Mac/CPU e Windows/GPU)
        self._check_hardware()

    def _check_hardware(self):
        """
        Verifica a disponibilidade de GPU e atualiza a configuração do modelo dinamicamente.
        Isso garante que o código corra tanto no Mac (CPU) quanto no Windows (GPU NVIDIA).
        """
        try:
            # Tenta importar uma biblioteca que verifique a GPU (ex: torch ou verificações do sistema)
            # Para fins do orquestrador, vamos simular a deteção ou verificar uma variável de ambiente
            use_gpu = os.environ.get('USE_GPU', 'False') == 'True'
            
            if use_gpu:
                logging.info("[HARDWARE] Acelerador GPU detectado. Otimização ativada.")
                self.config['model_training']['static_params']['task_type'] = 'GPU'
            else:
                logging.info("[HARDWARE] Nenhum acelerador GPU detectado. Fallback para processamento em CPU.")
                self.config['model_training']['static_params']['task_type'] = 'CPU'
        except Exception as e:
            logging.warning(f"[HARDWARE] Erro ao verificar hardware. Assumindo CPU. Erro: {e}")
            self.config['model_training']['static_params']['task_type'] = 'CPU'

    def step_1_ingestion(self):
        logging.info(">>> PASSO 1: Ingestão de Dados")
        path = self.config['data_paths']['raw_data']
        logging.info(f"Lendo dados de {path}...")
        # self.df = read_parquet(path)
        # Simulação para o Orquestrador
        logging.info("Ingestão concluída com sucesso.")

    def step_2_feature_engineering(self):
        logging.info(">>> PASSO 2: Engenharia de Features")
        engineer = FeatureEngineer(self.df)
        self.df = engineer.pipeline_completa(self.config['features'])
        logging.info(f"Aplicando Yeo-Johnson nas colunas: {self.df.columns}")
        # self.df = apply_yeo_johnson(self.df, criticas)
        
        radar = self.config['features']['radar_mahalanobis']
        logging.info(f"Calculando Mahalanobis para radar de anomalias nas colunas: {radar}")
        # self.df = detect_anomalies(self.df, radar)
        logging.info("Engenharia de Features concluída.")

    def step_3_model_training(self):
        logging.info(">>> PASSO 3: Treinamento do Modelo")
        tuning_config = self.config['model_training']['hyperparameter_tuning']
        
        hardware_mode = self.config['model_training']['static_params'].get('task_type', 'CPU')
        logging.info(f"Preparando treinamento. Modo de processamento: {hardware_mode}")
        
        if tuning_config['auto_tune']:
            logging.info(f"Modo Auto-Tune ATIVADO. Iniciando {tuning_config['engine'].upper()} com {tuning_config['n_trials']} trials.")
            logging.info(f"Otimizando métrica: {tuning_config['metric_to_optimize']}")
            # self.model = train_model_with_optuna(self.df, self.config)
        else:
            logging.info("Modo Auto-Tune DESATIVADO. Treinando com parâmetros estáticos.")
            # self.model = train_static_model(self.df, self.config)
        
        logging.info("Treinamento concluído.")

    def run_pipeline(self):
        start_time = datetime.now()
        logging.info(f"=== INICIANDO EXECUÇÃO DA PIPELINE: {self.config['pipeline_name']} ===")
        
        try:
            self.step_1_ingestion()
            self.step_2_feature_engineering()
            self.step_3_model_training()
            
            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"=== PIPELINE FINALIZADA COM SUCESSO. Duração: {duration} ===")
            
        except Exception as e:
            logging.error(f"!!! ERRO FATAL NA PIPELINE !!! Detalhes: {str(e)}")
            raise

if __name__ == "__main__":
    # Ponto de entrada da aplicação
    # Basta rodar: python pipeline_orchestrator.py
    pipeline = CreditRiskPipeline("config.json")
    pipeline.run_pipeline()