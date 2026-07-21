import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger("MLOps-DataLoader")

class DataLoader:
    """
    Classe responsável pela ingestão de dados de múltiplas fontes 
    e preparação dos conjuntos de dados (Train/Test Split OOT).
    """
    def __init__(self, config_paths: dict):
        self.paths = config_paths
        
    def _resolve_path(self, raw_path: str) -> str:
        """Resolve caminhos relativos em relação à raiz do projeto."""
        if raw_path.startswith("../"):
            # Assume que estamos a rodar a partir do orchestrator no backend
            base_dir = os.path.dirname(os.path.dirname(__file__)) # Sobe dois níveis
            return os.path.normpath(os.path.join(base_dir, raw_path.replace("../src/ifrs9_framework/", "")))
        return raw_path

    def ingest_data(self) -> pd.DataFrame:
        """
        Lê a base de dados configurada no YAML. 
        Preparado para expansão futura (ex: conector Snowflake).
        """
        raw_path = self.paths.get('raw_data', '')
        resolved_path = self._resolve_path(raw_path)
        
        logger.info(f"A iniciar ingestão de dados da fonte: {resolved_path}")
        
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Base de dados não encontrada no caminho: {resolved_path}")
            
        # Determina o motor de leitura com base na extensão
        if resolved_path.endswith('.parquet'):
            df = pd.read_parquet(resolved_path)
        elif resolved_path.endswith('.csv'):
            df = pd.read_csv(resolved_path)
        else:
            raise ValueError("Formato de ficheiro não suportado. Use .parquet ou .csv.")
            
        logger.info(f"Ingestão concluída. {len(df)} registos carregados.")
        return df

    def prepare_and_split(self, df: pd.DataFrame, target_col: str, feature_cols: list, test_size: float = 0.2):
        """
        Garante a separação cronológica (Out-Of-Time) dos dados.
        Vital para modelos IFRS9 e Risco de Crédito.
        """
        logger.info("A preparar partição de dados Out-Of-Time (OOT)...")
        
        # 1. Garantir ordem cronológica se a data existir
        if 'data_contratacao' in df.columns:
            df = df.sort_values('data_contratacao').reset_index(drop=True)
            logger.info("Base ordenada cronologicamente por 'data_contratacao'.")
            
        # 2. Filtrar apenas as colunas necessárias (Receita do Modelo)
        # Salvamos o target separadamente para não o perder
        if target_col not in df.columns:
            raise KeyError(f"A coluna target '{target_col}' não existe no dataframe.")
            
        y = df[target_col]
        
        # Seleciona as features (ignorando silenciosamente as que não existirem na base)
        cols_presentes = [c for c in feature_cols if c in df.columns]
        X = df[cols_presentes]
        
        # 3. Partição OOT (Sem Shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            shuffle=False # Regra de ouro para crédito!
        )
        
        logger.info(f"Partição concluída. Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas.")
        
        return X_train, X_test, y_train, y_test