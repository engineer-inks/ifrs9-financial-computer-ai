import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import shap
import logging

logger = logging.getLogger("MLOps-Auditoria")

class MetricsGenerator:
    """
    Motor analítico que avalia o modelo recém-treinado, aplica explicabilidade (SHAP)
    e gera o ficheiro JSON consumido pelo Painel Web.
    """
    def __init__(self, model, X_test, y_test, df_test_completo=None, cutoff=0.04):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.df_test = df_test_completo
        self.cutoff = cutoff

    def generate_metrics(self, output_path):
        logger.info("A iniciar cálculos de auditoria de risco...")
        
        # 1. Previsões e Probabilidades
        preds_proba = self.model.predict_proba(self.X_test)[:, 1]
        preds_class = (preds_proba > self.cutoff).astype(int)

        # 2. ROC e AUC
        fpr, tpr, _ = roc_curve(self.y_test, preds_proba)
        roc_auc = auc(fpr, tpr)

        # 3. Estatística KS
        ks_stat = np.max(tpr - fpr) * 100

        # 4. Matriz de Confusão e F1-Score
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds_class).ravel()
        f1 = f1_score(self.y_test, preds_class)

        # 5. SHAP (Feature Importance de Caixa-Branca)
        logger.info("A calcular SHAP values (Isto pode demorar alguns segundos)...")
        explainer = shap.TreeExplainer(self.model)
        # Usamos uma amostra se a base for muito grande para não travar o servidor
        amostra_shap = self.X_test.sample(min(10000, len(self.X_test)), random_state=42)
        shap_values = explainer.shap_values(amostra_shap)
        
        # Média absoluta do impacto de cada variável
        shap_abs = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': shap_abs
        }).sort_values('importance', ascending=False).head(10) # Top 10

        # 6. Backtesting OOT (Simulação de Safras)
        cohorts, ks_trend, auc_trend = [], [], []
        
        # Simulamos a divisão em safras quebrando a base de teste em 6 blocos cronológicos
        # (Assumindo que a base de teste já está ordenada cronologicamente)
        blocos = np.array_split(range(len(self.y_test)), 6)
        meses_mock = ["Mês 1", "Mês 2", "Mês 3", "Mês 4", "Mês 5", "Mês 6"]
        
        for i, idx_array in enumerate(blocos):
            if len(idx_array) > 0:
                y_true_bloco = self.y_test.iloc[idx_array]
                y_prob_bloco = preds_proba[idx_array]
                
                # Só calcula se houver caloteiros no bloco (para evitar erro de divisão)
                if len(np.unique(y_true_bloco)) > 1:
                    f, t, _ = roc_curve(y_true_bloco, y_prob_bloco)
                    cohorts.append(meses_mock[i])
                    auc_trend.append(float(auc(f, t)))
                    ks_trend.append(float(np.max(t - f) * 100))

        # 7. Montando o Pacote JSON
        logger.info("A empacotar métricas para o Dashboard Web...")
        metrics_dict = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "kpis": {
                "ks": float(ks_stat),
                "auc": float(roc_auc),
                "f1": float(f1),
                "saved": int(13151 - fp) # Quantos bons clientes salvámos em relação ao modelo original ruim
            },
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
            "roc_curve": {
                # Fazemos um slice [::5] para reduzir o tamanho do ficheiro (otimização web)
                "fpr": fpr[::5].tolist(), 
                "tpr": tpr[::5].tolist()
            },
            "feature_importance": {
                "features": feature_importance['feature'].tolist(),
                "scores": feature_importance['importance'].tolist()
            },
            "backtest": {
                "cohorts": cohorts,
                "ks_trend": ks_trend,
                "auc_trend": auc_trend
            }
        }

        # Guardar no disco
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f)
            
        logger.info(f"Métricas gravadas com sucesso em: {output_path}")
        return metrics_dict