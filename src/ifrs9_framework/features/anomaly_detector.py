import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import chi2
import logging

logger = logging.getLogger("MLOps-AnomalyDetector")

class AnomalyDetector:
    """
    Módulo especializado na deteção de anomalias multivariadas em dados de crédito.
    Utiliza a Distância de Mahalanobis para identificar perfis de clientes que,
    embora normais em variáveis isoladas, apresentam combinações altamente suspeitas.
    """
    
    def __init__(self, confidence_level: float = 0.995):
        """
        :param confidence_level: Nível de confiança estatística para o corte (default: 99.5%).
                                 Valores acima deste limiar na distribuição Chi-Quadrado 
                                 são considerados anomalias.
        """
        self.confidence_level = confidence_level
        self.inv_covmat = None
        self.mean_dist = None
        self.features_used = None

    def _calculate_mahalanobis(self, data: np.ndarray) -> np.ndarray:
        """
        Calcula a Distância de Mahalanobis de forma vetorizada (muito mais rápida).
        """
        # Centraliza os dados
        y_mu = data - self.mean_dist
        
        # Distância = (x - mu) * Cov^-1 * (x - mu)^T
        left_term = np.dot(y_mu, self.inv_covmat)
        mahal_distances = np.sum(left_term * y_mu, axis=1)
        
        return mahal_distances

    def fit(self, df: pd.DataFrame, feature_cols: list):
        """
        Aprende o padrão de 'normalidade' da base de dados de treino calculando
        a matriz de covariância e a média multidimensional.
        """
        logger.info(f"Treinando o Radar de Anomalias (Mahalanobis) nas colunas: {feature_cols}")
        
        self.features_used = [c for c in feature_cols if c in df.columns]
        if len(self.features_used) < 2:
            raise ValueError("O Detector de Anomalias precisa de pelo menos 2 variáveis numéricas.")

        data = df[self.features_used].values
        
        # 1. Calcula o centro de massa (média)
        self.mean_dist = np.mean(data, axis=0)
        
        # 2. Calcula a matriz de covariância
        cov_matrix = np.cov(data, rowvar=False)
        
        # 3. Calcula a matriz inversa (usamos pseudo-inversa por segurança contra colinearidade)
        self.inv_covmat = sp.linalg.pinv(cov_matrix)
        
        logger.info("Radar calibrado com sucesso.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o radar à base de dados. Adiciona a coluna de distância 
        e a flag binária de anomalia baseada na distribuição Chi-Square.
        """
        if self.inv_covmat is None:
            raise RuntimeError("O modelo precisa ser treinado (fit) antes de usar o transform.")
            
        logger.info("Aplicando a deteção de anomalias na base...")
        df_out = df.copy()
        
        # Extrai os valores das colunas aprendidas
        data = df_out[self.features_used].values
        
        # Calcula as distâncias
        df_out['distancia_mahalanobis'] = self._calculate_mahalanobis(data)
        
        # Calcula o Limiar Dinâmico (Chi-Square)
        # Graus de liberdade (df) = número de variáveis no radar
        degrees_of_freedom = len(self.features_used)
        threshold = chi2.ppf(self.confidence_level, degrees_of_freedom)
        
        # Aplica a flag (1 = Anomalia / Fraude Potencial, 0 = Normal)
        df_out['anomalia_multivariada'] = (df_out['distancia_mahalanobis'] > threshold).astype(int)
        
        qtd_anomalias = df_out['anomalia_multivariada'].sum()
        pct_anomalias = (qtd_anomalias / len(df_out)) * 100
        
        logger.info(f"Limiar de Corte (Chi2 {self.confidence_level*100}%): {threshold:.2f}")
        logger.info(f"Anomalias detetadas: {qtd_anomalias} contratos ({pct_anomalias:.2f}% da base).")
        
        return df_out

    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Executa treino e transformação numa única chamada."""
        self.fit(df, feature_cols)
        return self.transform(df)