import os
import gc
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import catboost as cb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, recall_score, precision_score

from advanced_metrics_and_split import otimizar_pesos_e_threshold_f1

PRETO = "#000000"
AZUL_PRIMARIO = "#1526FF"
AZUL_SECUNDARIO = "#0066FC"
ROSA = "#FF007F"
CINZA_CLARO = "#CCCCCC"

logger = logging.getLogger("IFRS9_Engine.OOF_Calib")

def calcular_metricas_performance(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, mcc

def gerar_grafico_resumo_kpis_completo(auc, ks, acc, mcc, f1, prec, rec, output_dir):
    kpis = ['AUC', 'KS', 'Acurácia', 'MMC', 'F1-Score', 'Precision', 'Recall']
    valores = [auc, ks, acc, mcc, f1, prec, rec]
    
    plt.figure(figsize=(12, 8), facecolor=PRETO)
    ax = plt.gca()
    ax.set_facecolor(PRETO)
    
    colors = [AZUL_PRIMARIO, AZUL_SECUNDARIO, ROSA, AZUL_PRIMARIO, AZUL_SECUNDARIO, ROSA, AZUL_PRIMARIO]
    
    bars = plt.barh(kpis, valores, color=colors)
    plt.bar_label(bars, fmt='%.4f', padding=10, color='white', fontweight='bold', fontsize=11)
    
    plt.title('Dashboard de Performance do Modelo (Métricas IFRS9)', color='white', fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, 1.2)
    ax.tick_params(colors=CINZA_CLARO, labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'resumo_kpis_completo.png'), facecolor=PRETO, dpi=130)
    plt.close()

def aplicar_calibracao_hibrida(prob_bruta, calibrator_model, weight, method_type):
    prob_shift = prob_bruta / (prob_bruta + (1 - prob_bruta) * weight)
    
    if calibrator_model is not None:
        prob_2d = prob_shift.reshape(-1, 1)
        if method_type == 'isotonic':
            prob_final = calibrator_model.transform(prob_shift)
        elif method_type == 'beta':
            prob_final = calibrator_model.predict(prob_shift)
        elif method_type == 'spline':
            prob_final = calibrator_model.predict_proba(prob_2d)[:, 1]
        else:
            prob_final = prob_shift
        return np.clip(prob_final, 0, 1)
    return np.clip(prob_shift, 0, 1)

def optimization_threshold(config, model_final, X_train_val_processed, y_train_val, groups_train_val, 
                           melhores_params, X_test_processed, y_test, cat_indices, preprocessor, 
                           sample=True, method='isotonic'):
    logger.info("="*80)
    logger.info(f"--- PIPELINE DE CALIBRAÇÃO v82 (Modo: {'FULL' if sample else 'FAST'}) | Método: {method.upper()} ---")
    logger.info("="*80)
    
    w_train = melhores_params.get('w_train_calculado', melhores_params.get('scale_pos_weight', 1.0))
    logger.info(f"⚙️ Peso de treino: {w_train:.4f}. Aplicando neutralização OOF.")
    
    if sample:
        n_splits = 5
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.get('random_state', 42))
        oof_probs_raw = np.zeros(len(y_train_val))
        
        if not isinstance(X_train_val_processed, np.ndarray):
            X_train_val_processed = X_train_val_processed.to_numpy()
        if hasattr(y_train_val, 'values'):
            y_train_arr = y_train_val.values
        else:
            y_train_arr = y_train_val
            
        for i, (train_idx, val_idx) in enumerate(cv.split(X_train_val_processed, y_train_arr, groups=groups_train_val)):
            X_tr, X_hold = X_train_val_processed[train_idx], X_train_val_processed[val_idx]
            y_tr = y_train_arr[train_idx]
            
            model_fold = cb.CatBoostClassifier(**melhores_params, verbose=0, thread_count=-1, allow_writing_files=False)
            model_fold.fit(X_tr, y_tr, cat_features=cat_indices)
            
            oof_probs_raw[val_idx] = model_fold.predict_proba(X_hold)[:, 1]
            del model_fold; gc.collect()
            
        y_calib_true = y_train_arr
    else:
        logger.info("Executando Calibração Rápida...")
        oof_probs_raw = model_final.predict_proba(X_train_val_processed)[:, 1]
        y_calib_true = y_train_val
        
    oof_probs_neutral = aplicar_calibracao_hibrida(oof_probs_raw, None, w_train, method)
    calibrator = None
    
    if method == 'isotonic':
        logger.info("Ajustando Isotonic Regression...")
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        calibrator.fit(oof_probs_neutral, y_calib_true)
    elif method == 'beta':
        logger.info("Ajustando Beta Calibration...")
        try:
            from betacal import BetaCalibration
            calibrator = BetaCalibration(parameters="abm")
            calibrator.fit(oof_probs_neutral, y_calib_true)
        except Exception as e:
            logger.error(f'Falha no BetaCalibration: {e}.')
    elif method == 'spline':
        logger.info("Ajustando Spline Calibration (Cubic)...")
        calibrator = Pipeline([
            ('spline', SplineTransformer(n_knots=5, degree=3, knots='quantile')),
            ('logistic', LogisticRegression(penalty=None))
        ])
        calibrator.fit(oof_probs_neutral.reshape(-1, 1), y_calib_true)

    y_oof_calibrated = aplicar_calibracao_hibrida(oof_probs_raw, calibrator, w_train, method)
    
    resultado_otimo = otimizar_pesos_e_threshold_f1(
        y_true=y_calib_true, 
        y_prob_raw=y_oof_calibrated, 
        output_dir=config['GRAPHICS_DIR'], 
        w_train=1.0
    )
    
    threshold_otimo = resultado_otimo['threshold']
    peso_grid = resultado_otimo['peso']
    
    y_prob_test_raw = model_final.predict_proba(X_test_processed)[:, 1]
    y_prob_test_calib = aplicar_calibracao_hibrida(y_prob_test_raw, calibrator, w_train, method)
    
    y_prob_test_final = y_prob_test_calib / (y_prob_test_calib + (1 - y_prob_test_calib) / peso_grid)
    y_prob_test_final = np.clip(y_prob_test_final, 0, 1)
    
    y_pred_test = (y_prob_test_final >= threshold_otimo).astype(int)
    f1_h = f1_score(y_test, y_pred_test, zero_division=0)
    
    logger.info("="*80)
    logger.info(f"🏆 PERFORMANCE FINAL ({method.upper()}) - Ratio: {y_prob_test_final.mean() / (y_test.mean() + 1e-9):.2f}x")
    logger.info(f"   F1-Score: {f1_h:.4f}")
    logger.info("="*80)
    
    return calibrator, resultado_otimo, f1_h, y_prob_test_final, y_pred_test