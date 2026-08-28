import os
import gc
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, recall_score, precision_score, accuracy_score, matthews_corrcoef, confusion_matrix

logger = logging.getLogger("IFRS9_Engine.Evaluation")

def gerar_metricas_classificacao_logs(y_true, y_prob, threshold, method='modelo'):
    t_corte = threshold['threshold'] if isinstance(threshold, dict) else threshold
    y_pred = (y_prob >= t_corte).astype(int)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ratio_classif = y_pred.mean() / (y_true.mean() + 1e-9)
    
    logger.info("================================================================================")
    logger.info(f"--- MATRIZ DE CONFUSÃO - MÉTODO: {method.upper()} ---")
    logger.info(f"--- PONTO DE CORTE OTIMIZADO: {t_corte*100:.2f}% ---")
    logger.info("-" * 80)
    logger.info(f"BONS Reais (0): {tn+fp:<10,} | Acertos: {tn:<10,} (TN) | Erros: {fp:<10,} (FP)")
    logger.info(f"MAUS Reais (1): {fn+tp:<10,} | Acertos: {tp:<10,} (TP) | Erros: {fn:<10,} (FN)")
    logger.info("-" * 80)
    logger.info(f"📊 PERFORMANCE: ACC: {acc:.4f} | MCC: {mcc:.4f} | Ratio: {ratio_classif:.2f}x")
    logger.info(f"📈 DECISÃO:     PRECISION: {precision:.4f} | RECALL: {recall:.4f} | F1: {f1:.4f}")
    logger.info("================================================================================")
    
    return {
        'metodo': method, 'precision': precision, 'recall': recall, 'f1_score': f1,
        'ratio_classif': ratio_classif, 'threshold_usado': t_corte, 'acc': acc, 'mcc': mcc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def executar_teste_aderencia_buckets(y_true, y_prob, n_bins=10):
    df_calib = pd.DataFrame({'y_true': np.array(y_true), 'y_prob': np.array(y_prob)})
    try:
        df_calib['bucket'] = pd.qcut(df_calib['y_prob'], q=n_bins, duplicates='drop')
    except Exception:
        df_calib['bucket'] = pd.cut(df_calib['y_prob'], bins=n_bins)
        
    stats_bucket = df_calib.groupby('bucket', observed=False).agg(
        N=('y_true', 'count'), Observed_Def=('y_true', 'sum'), Prob_Mean=('y_prob', 'mean')
    ).reset_index()
    
    stats_bucket['Observed_Rate'] = stats_bucket['Observed_Def'] / stats_bucket['N']
    z_score = 1.96
    p_model = stats_bucket['Prob_Mean']
    erro_padrao = z_score * np.sqrt((p_model * (1 - p_model)) / stats_bucket['N'].replace(0,1))
    stats_bucket['IC_Lower'] = (p_model - erro_padrao).clip(lower=0)
    stats_bucket['IC_Upper'] = (p_model + erro_padrao).clip(upper=1)
    
    rmse_bucket = np.sqrt(((stats_bucket['Observed_Rate'] - stats_bucket['Prob_Mean'])**2).mean())
    
    logger.info("================================================================================")
    logger.info("--- TESTE DE CALIBRAÇÃO POR FAIXA (TRAFFIC LIGHT) ---")
    logger.info(f"--- Erro Médio por Faixa (RMSE): {rmse_bucket:.4%} ---")
    logger.info("-" * 105)
    
    header = f"{'Faixa':<5} | {'N Clientes':<10} | {'PD Modelo':<9} | {'PD Real':<9} | {'IC Modelo [Min - Max]':<22} | {'Status'}"
    logger.info(header)
    logger.info("-" * 105)
    
    for i, row in stats_bucket.iterrows():
        real = row['Observed_Rate']
        ic_min = row['IC_Lower']
        ic_max = row['IC_Upper']
        
        if ic_min <= real <= ic_max:
            status_str = "[ VERDE   ]"
        elif abs(real - row['Prob_Mean']) < 0.05 * row['Prob_Mean']:
            status_str = "[ WARNING ]"
        else:
            status_str = "[ VERM    ]"
            
        log_func = logger.warning if "VERM" in status_str or "WARNING" in status_str else logger.info
        line = (f"{i+1:<5} | {int(row['N']):<10} | {row['Prob_Mean']:.2%}    | {row['Observed_Rate']:.2%}    | "
                f"[{row['IC_Lower']:.2%} - {row['IC_Upper']:.2%}]    | {status_str}")
        log_func(line)
        
    logger.info("================================================================================\n")
    return stats_bucket

def salvar_pipeline_completo(config, preprocessor, model, threshold, feature_names, cat_indices, scale_pos_weight, isotonic_model):
    algo_name = config.get('algorithm', 'catboost').lower()
    nome_arquivo = f"pipeline_{algo_name}_v1.joblib"
    
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    full_pipeline.calibration_meta_ = {
        'scale_pos_weight': scale_pos_weight, 'scalar_factor': 1.0,
        'isotonic_model': isotonic_model, 'threshold_corte': threshold,
        'feature_names': feature_names, 'cat_indices': cat_indices,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    }
    
    models_dir = config.get('models_dir') if isinstance(config, dict) else None
    if not models_dir:
        models_dir = config.get('data_paths', {}).get('model_output', 'models') if isinstance(config, dict) else 'models'
    
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, nome_arquivo)
    joblib.dump(full_pipeline, path)
    
    logger.info(f"✅ Pipeline Completo salvo em: {path}")
    logger.info(f"   -> Inclui Threshold de: {threshold:.4f}")