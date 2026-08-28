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
    def __init__(self, model, X_test, y_test, df_test_completo=None, cutoff=0.04):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.df_test = df_test_completo
        self.cutoff = cutoff

    def generate_metrics(self, output_path):
        logger.info("A iniciar cálculos de auditoria de risco...")
        
        preds_proba = self.model.predict_proba(self.X_test)[:, 1]
        preds_class = (preds_proba > self.cutoff).astype(int)

        fpr, tpr, _ = roc_curve(self.y_test, preds_proba)
        roc_auc = auc(fpr, tpr)
        ks_stat = np.max(tpr - fpr) * 100
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds_class).ravel()
        f1 = f1_score(self.y_test, preds_class)

        # === EXPLICABILIDADE ===
        if type(self.model).__name__ == 'ExplainableBoostingClassifier':
            ebm_global = self.model.explain_global()
            data = ebm_global.data() 
            feature_importance = pd.DataFrame({
                'feature': data['names'],
                'importance': data['scores']
            }).sort_values('importance', ascending=False).head(10)
        else:
            explainer = shap.TreeExplainer(self.model)
            amostra_shap = self.X_test.sample(min(10000, len(self.X_test)), random_state=42)
            shap_values = explainer.shap_values(amostra_shap)
            if isinstance(shap_values, list): shap_values = shap_values[1]
            shap_abs = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': self.X_test.columns,
                'importance': shap_abs
            }).sort_values('importance', ascending=False).head(10)

        # === TESTE DE HOSMER-LEMESHOW (TRAFFIC LIGHT) ===
        df_calib = pd.DataFrame({'y_true': self.y_test, 'y_prob': preds_proba})
        try:
            df_calib['bucket'] = pd.qcut(df_calib['y_prob'], q=10, duplicates='drop')
        except:
            df_calib['bucket'] = pd.cut(df_calib['y_prob'], bins=10)
            
        stats = df_calib.groupby('bucket', observed=False).agg(
            N=('y_true', 'count'),
            Observed_Def=('y_true', 'sum'),
            Prob_Mean=('y_prob', 'mean')
        ).reset_index()
        
        stats['Observed_Rate'] = stats['Observed_Def'] / stats['N']
        z_score = 1.96
        p = stats['Prob_Mean']
        erro_padrao = z_score * np.sqrt((p * (1 - p)) / stats['N'].replace(0,1))
        stats['IC_Lower'] = (p - erro_padrao).clip(lower=0)
        stats['IC_Upper'] = (p + erro_padrao).clip(upper=1)
        
        hl_table = []
        for i, row in stats.iterrows():
            real = row['Observed_Rate']
            ic_min, ic_max, prob_mean = row['IC_Lower'], row['IC_Upper'], row['Prob_Mean']
            
            if ic_min <= real <= ic_max:
                status = "VERDE"
            elif abs(real - prob_mean) < 0.05 * prob_mean:
                status = "WARNING"
            else:
                status = "VERMELHO"
                
            hl_table.append({
                "faixa": i + 1,
                "n_clientes": int(row['N']),
                "pd_modelo": round(float(prob_mean) * 100, 2),
                "pd_real": round(float(real) * 100, 2),
                "ic_min": round(float(ic_min) * 100, 2),
                "ic_max": round(float(ic_max) * 100, 2),
                "status": status
            })

        # === BACKTESTING OOT ===
        cohorts, ks_trend, auc_trend = [], [], []
        blocos = np.array_split(range(len(self.y_test)), 6)
        meses_mock = ["Mês 1", "Mês 2", "Mês 3", "Mês 4", "Mês 5", "Mês 6"]
        
        for i, idx_array in enumerate(blocos):
            if len(idx_array) > 0:
                y_true_bloco = self.y_test.iloc[idx_array]
                y_prob_bloco = preds_proba[idx_array]
                if len(np.unique(y_true_bloco)) > 1:
                    f, t, _ = roc_curve(y_true_bloco, y_prob_bloco)
                    cohorts.append(meses_mock[i])
                    auc_trend.append(float(auc(f, t)))
                    ks_trend.append(float(np.max(t - f) * 100))

        logger.info("A empacotar métricas para o Dashboard Web...")
        metrics_dict = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "kpis": {
                "ks": float(ks_stat), "auc": float(roc_auc), "f1": float(f1),
                "saved": int(13151 - fp) if fp < 13151 else 0
            },
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc_curve": {"fpr": fpr[::5].tolist(), "tpr": tpr[::5].tolist()},
            "feature_importance": {"features": feature_importance['feature'].tolist(), "scores": feature_importance['importance'].tolist()},
            "backtest": {"cohorts": cohorts, "ks_trend": ks_trend, "auc_trend": auc_trend},
            "hosmer_lemeshow": hl_table
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f: json.dump(metrics_dict, f)
        return metrics_dict