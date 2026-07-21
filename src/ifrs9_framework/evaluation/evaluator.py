import pandas as pd
import numpy as np
import shap
import logging

logger = logging.getLogger("MLOps-Explainer")

class ModelExplainer:
    """
    Módulo responsável por desconstruir o modelo e extrair a importância das variáveis.
    Tem inteligência para distinguir entre modelos Caixa-Branca (EBM) e Caixa-Preta (CatBoost/LGBM).
    """
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test

    def get_feature_importance(self, top_n: int = 10) -> dict:
        """
        Retorna as features mais importantes num dicionário pronto para a UI HTML.
        :param top_n: Quantas variáveis queremos devolver para o gráfico de barras.
        """
        algo_name = type(self.model).__name__
        
        if algo_name == 'ExplainableBoostingClassifier':
            logger.info("[EXPLICAÇÃO] Modelo EBM detectado. A extrair regras nativas (Global Explain)...")
            return self._extract_ebm_importance(top_n)
        else:
            logger.info(f"[EXPLICAÇÃO] Modelo {algo_name} detectado. A calcular SHAP values...")
            return self._extract_shap_importance(top_n)

    def _extract_ebm_importance(self, top_n: int) -> dict:
        """Lógica exclusiva para extrair o peso exato da biblioteca InterpretML (Microsoft)"""
        ebm_global = self.model.explain_global()
        data = ebm_global.data()
        
        df_imp = pd.DataFrame({
            'feature': data['names'],
            'importance': data['scores']
        })
        
        # Ordenamos do mais importante para o menos e limitamos aos top_n
        df_imp = df_imp.sort_values('importance', ascending=False).head(top_n)
        
        return {
            "features": df_imp['feature'].tolist(),
            "scores": df_imp['importance'].tolist()
        }

    def _extract_shap_importance(self, top_n: int) -> dict:
        """Lógica universal usando SHAP TreeExplainer para florestas de decisão"""
        explainer = shap.TreeExplainer(self.model)
        
        # Amostragem para evitar lentidão extrema na pipeline se a base tiver milhões de linhas
        amostra_shap = self.X_test.sample(min(10000, len(self.X_test)), random_state=42)
        shap_values = explainer.shap_values(amostra_shap)
        
        # Dependendo do algoritmo/versão, os SHAP values do LightGBM podem vir numa lista.
        # Nós queremos a importância para a classe 1 (Calote).
        if isinstance(shap_values, list):
            shap_values = shap_values[1] 
            
        # O impacto global é a média absoluta do impacto individual de cada cliente
        shap_abs = np.abs(shap_values).mean(axis=0)
        
        df_imp = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': shap_abs
        })
        
        df_imp = df_imp.sort_values('importance', ascending=False).head(top_n)
        
        return {
            "features": df_imp['feature'].tolist(),
            "scores": df_imp['importance'].tolist()
        }