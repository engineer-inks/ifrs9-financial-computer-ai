import numpy as np
import pandas as pd
import os
import uuid
from datetime import datetime, timedelta

class StochasticTemporalDataGenerator:
    def __init__(self, n_clientes: int = 150000, random_state: int = 42):
        self.n_clientes = n_clientes
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.data_ref = datetime(2026, 6, 12)
        
    def generate_clients(self) -> pd.DataFrame:
        print(f"--- [1/3] Gerando {self.n_clientes} Clientes (Assimetria Pareto) ---")
        
        cliente_id = np.arange(1, self.n_clientes + 1)
        idade = np.random.normal(52, 14, self.n_clientes).clip(18, 95)
        
        renda_pareto = (np.random.pareto(a=1.16, size=self.n_clientes) + 1) * 1320
        renda = renda_pareto.clip(1320, 150000) 
        
        bureau_score = (np.random.normal(650, 120, self.n_clientes) + np.random.normal(0, 50, self.n_clientes)).clip(0, 1000)
        
        ufs = ['SP', 'MG', 'RJ', 'BA', 'RS', 'PR', 'PE', 'CE', 'PA', 'MA']
        uf_residencia = np.random.choice(ufs, size=self.n_clientes, p=[0.22, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04, 0.04, 0.28])
        
        zona_risco = np.random.binomial(1, np.where(renda < 3000, 0.15, 0.01))
        obito_flag = np.random.binomial(1, np.where(idade > 75, 0.05, np.where(idade > 60, 0.01, 0.001)))
        
        return pd.DataFrame({'cliente_id': cliente_id, 'idade': idade, 'renda': renda, 
                             'uf': uf_residencia, 'zona_risco_flag': zona_risco, 
                             'bureau_score': bureau_score, 'obito_flag': obito_flag})

    def generate_contracts(self, df_clientes: pd.DataFrame) -> pd.DataFrame:
        print("--- [2/3] Injetando Ruído Estocástico Multivariado nas Features ---")
        
        n_contratos = np.random.poisson(lam=1.5, size=self.n_clientes) + 1
        df_contratos = df_clientes.loc[df_clientes.index.repeat(n_contratos)].reset_index(drop=True)
        total_contratos = len(df_contratos)
        
        df_contratos['codigo_contrato'] = [str(uuid.uuid4())[:8] for _ in range(total_contratos)]
        produtos = ['Consignado INSS', 'Consignado SIAPE', 'Crédito Pessoal', 'Cartão de Crédito']
        df_contratos['tipo_produto'] = np.random.choice(produtos, size=total_contratos, p=[0.50, 0.20, 0.15, 0.15])
        
        dias_passados = np.random.randint(1, 1800, size=total_contratos)
        df_contratos['data_contratacao'] = [self.data_ref - timedelta(days=int(d)) for d in dias_passados]
        
        # 1. QUEBRA DE LINEARIDADE: Ruído nos Prazos
        # Em vez de 84 cravado, temos uma distribuição normal em volta de 84
        prazo_consignado = np.random.normal(80, 12, total_contratos).astype(int).clip(24, 120)
        prazo_pessoal = np.random.normal(30, 8, total_contratos).astype(int).clip(12, 60)
        df_contratos['prazo_meses'] = np.where(df_contratos['tipo_produto'].str.contains('Consignado'), prazo_consignado, prazo_pessoal)
        
        df_contratos['data_vencimento'] = df_contratos.apply(lambda row: row['data_contratacao'] + timedelta(days=int(row['prazo_meses'])*30), axis=1)
        
        # 2. QUEBRA DE LINEARIDADE: Limites Borrados (Fuzziness)
        # Multiplicador log-normal (cauda longa) + ruído puro
        base_limite = df_contratos['renda'] * np.random.lognormal(mean=0.2, sigma=0.6, size=total_contratos)
        ruido_limite = np.random.normal(0, 1000, total_contratos)
        limite_sujo = np.maximum(0, base_limite + ruido_limite).clip(0, 50000)
        
        # Simulando cartões bloqueados (5% dos cartões com limite 0) para misturar os clusters
        cartao_mask = df_contratos['tipo_produto'] == 'Cartão de Crédito'
        limite_bloqueado = np.random.choice([True, False], size=total_contratos, p=[0.05, 0.95])
        
        df_contratos['limite_credito'] = np.where(cartao_mask & limite_bloqueado, 0.0,
                                         np.where(cartao_mask, limite_sujo, 0.0))
        
        # 3. QUEBRA DE LINEARIDADE: Distribuição Gama para Valor Financiado
        # O scale (escala) é influenciado pela renda, mas o shape garante aleatoriedade orgânica
        scale_gama = df_contratos['renda'] * 1.5
        valor_financiado_gama = np.random.gamma(shape=2.5, scale=scale_gama)
        
        df_contratos['valor_financiado'] = np.where(
            df_contratos['tipo_produto'] == 'Cartão de Crédito',
            df_contratos['limite_credito'] * np.random.beta(a=1.5, b=5.0, size=total_contratos), # Uso rotativo
            valor_financiado_gama # Empréstimo via Gama
        )
        
        df_contratos['selic_contratacao'] = np.random.uniform(2.0, 13.75, total_contratos)
        
        return df_contratos

    def apply_latent_risk_and_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        print("--- [3/3] Calculando Sobrevivência: Churn e Default (Meta: ~2%) ---")
        
        z = -4.8 - 0.001 * df['bureau_score'] + 0.1 * (df['selic_contratacao'] / 10)
        
        comprometimento = df['valor_financiado'] / (df['renda'] * 12)
        z += np.where(comprometimento > 0.4, 0.8, 0.0) 
        z += np.where(df['zona_risco_flag'] == 1, 0.5, 0.0)
        z += np.where(df['obito_flag'] == 1, 6.0, 0.0) 
        
        z += np.where(df['tipo_produto'].isin(['Cartão de Crédito', 'Crédito Pessoal']), 2.0, -1.0)
        
        prob_default = 1 / (1 + np.exp(-z))
        df['default_flag'] = (prob_default > np.random.uniform(0, 1, len(df))).astype(int)
        
        prob_churn = np.where(df['tipo_produto'].str.contains('Consignado'), 0.15, 0.05)
        df['churn_flag'] = np.where(df['default_flag'] == 1, 0, (np.random.uniform(0, 1, len(df)) < prob_churn).astype(int))
        
        meses_ativos = (self.data_ref - df['data_contratacao']).dt.days // 30
        meses_ate_evento = np.random.randint(1, np.maximum(meses_ativos + 1, 2))
        
        df['data_evento'] = df.apply(lambda row: row['data_contratacao'] + timedelta(days=int(row['prazo_meses'])*30) 
                                     if row['default_flag'] == 0 and row['churn_flag'] == 0 
                                     else row['data_contratacao'] + timedelta(days=int(meses_ate_evento[row.name])*30), axis=1)

        print(f"Total de Contratos: {len(df)}")
        print(f"Taxa Real de Default: {df['default_flag'].mean() * 100:.2f}%")
        print(f"Taxa Real de Churn: {df['churn_flag'].mean() * 100:.2f}%\n")
        
        return df

    def run_and_save(self, output_path: str):
        df = self.generate_clients()
        df = self.generate_contracts(df)
        df = self.apply_latent_risk_and_churn(df)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Salvando base Estocástica em {output_path}...")
        df.to_parquet(output_path, index=False)
        print("Finalizado com sucesso!")

if __name__ == "__main__":
    generator = StochasticTemporalDataGenerator(n_clientes=150000)
    generator.run_and_save("/workspace/src/ifrs9_framework/data/raw/synthetic_credit_data.parquet")