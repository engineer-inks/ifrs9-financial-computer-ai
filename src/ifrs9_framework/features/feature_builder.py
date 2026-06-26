import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import PowerTransformer

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Módulo responsável por criar novas variáveis (Feature Engineering)
    focadas em risco de crédito e comportamento financeiro.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def criar_features_comportamentais(self):
        """
        Cria rácios financeiros que expõem a alavancagem e o comprometimento 
        do cliente, ajudando o modelo a identificar fraudes sintéticas.
        """
        logger.info("Criando Features Comportamentais e Financeiras...")
        
        # 1. Comprometimento de Renda (Rácio Financiamento / Renda Mensal)
        # Proteção contra divisão por zero
        renda_segura = np.where(self.df['renda'] == 0, 1, self.df['renda'])
        self.df['ratio_financiamento_renda'] = self.df['valor_financiado'] / renda_segura
        
        # 2. Alavancagem de Crédito (Limite do Cartão / Renda)
        self.df['ratio_limite_renda'] = self.df['limite_credito'] / renda_segura
        
        # 3. Fator Idade-Renda (Identifica jovens com rendas suspeitamente altas)
        idade_segura = np.where(self.df['idade'] == 0, 1, self.df['idade'])
        self.df['fator_idade_renda'] = self.df['renda'] / idade_segura
        
        # 4. Intensidade da Parcela Simulada (Valor Financiado / Prazo) / Renda
        prazo_seguro = np.where(self.df['prazo_meses'] == 0, 1, self.df['prazo_meses'])
        parcela_estimada = self.df['valor_financiado'] / prazo_seguro
        self.df['impacto_parcela_renda'] = parcela_estimada / renda_segura

        logger.info("Rácios financeiros gerados com sucesso.")
        return self.df

    def aplicar_transformacoes_matematicas(self, colunas_criticas: list):
        """
        Aplica a transformação Yeo-Johnson para estabilizar a variância 
        de caudas longas (outliers) transformando-as em distribuições normais.
        """
        logger.info(f"Aplicando Transformação Yeo-Johnson nas colunas: {colunas_criticas}")
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        
        for col in colunas_criticas:
            nome_nova_col = f"{col}_yj"
            self.df[nome_nova_col] = pt.fit_transform(self.df[[col]])
            
        logger.info("Transformações matemáticas concluídas.")
        return self.df

    def pipeline_completa(self, config_features: dict):
        """Executa toda a esteira de features em sequência."""
        self.criar_features_comportamentais()
        
        if 'critical_for_yeo_johnson' in config_features:
            colunas_yj = config_features['critical_for_yeo_johnson']
            # Adicionamos as novas features de rácio para normalização também
            colunas_yj.extend(['ratio_financiamento_renda', 'impacto_parcela_renda'])
            self.aplicar_transformacoes_matematicas(colunas_yj)
            
        return self.df