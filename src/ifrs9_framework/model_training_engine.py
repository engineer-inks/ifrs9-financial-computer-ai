import numpy as np
import pandas as pd
import logging
import gc
import traceback
import catboost as cb
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss

logger = logging.getLogger("IFRS9_Engine.Training")

def preparar_matrizes(df_train_val, df_test, config, preprocessor_func, tracker=None):
    target = config.get('target_column', 'default_flag')
    group_col = config.get('group_column', 'codigo_contrato')
    cols_to_drop_config = config.get('columns_to_drop', [])
    
    def log_and_track(msg):
        logger.info(msg)
        if tracker: tracker.update_node("step_2", "running", msg)

    def split_X_y_raw(df_subset):
        y_out = df_subset[target].copy()
        cols_to_drop = [target] + [c for c in cols_to_drop_config if c in df_subset.columns]
        X_out = df_subset.drop(columns=cols_to_drop)
        return X_out, y_out
        
    log_and_track("Separando matrizes X e y...")
    X_train_val_raw, y_train_val = split_X_y_raw(df_train_val)
    X_test_raw, y_test = split_X_y_raw(df_test)
    
    # 2. Identifica colunas constantes (VETORIZADO = RÁPIDO)
    log_and_track("Identificando colunas constantes (método vetorizado)...")
    nunique_vals = X_train_val_raw.nunique(dropna=False)
    cols_constantes = nunique_vals[nunique_vals <= 1].index.tolist()
    
    if cols_constantes:
        log_and_track(f"Removendo {len(cols_constantes)} colunas constantes.")
        X_train_val_raw.drop(columns=cols_constantes, inplace=True)
        X_test_raw.drop(columns=cols_constantes, inplace=True, errors='ignore')
        
    groups_train_val = df_train_val[group_col].values
    log_and_track(f"Realizando Split Treino/Validação por Grupos ({group_col})...")
    
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.get('random_state', 42))
    tr_idx, val_idx = next(splitter_val.split(X_train_val_raw, y_train_val, groups=groups_train_val))
    
    X_train_raw = X_train_val_raw.iloc[tr_idx]
    y_train = y_train_val.iloc[tr_idx]
    X_val_raw = X_train_val_raw.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]
    
    log_and_track(f"Treino: {X_train_raw.shape[0]:,} | Val: {X_val_raw.shape[0]:,} | Teste: {X_test_raw.shape[0]:,}")
    
    preprocessor = preprocessor_func(X_train_raw, config, logger)
    log_and_track("Aplicando pré-processamento nas matrizes (fit_transform)...")
    
    # GESTÃO AGRESSIVA DE MEMÓRIA (Evita OOM Kill)
    try:
        X_train_processed = preprocessor.fit_transform(X_train_raw)
        del X_train_raw
        gc.collect() # Força libertação de RAM imediata
        
        X_val_processed = preprocessor.transform(X_val_raw)
        del X_val_raw
        gc.collect()
        
        X_test_processed = preprocessor.transform(X_test_raw)
        del X_test_raw
        gc.collect()
        
        X_train_val_processed = preprocessor.transform(X_train_val_raw)
        del X_train_val_raw
        gc.collect()
    except Exception as e:
        err_msg = f"CRASH NO PRÉ-PROCESSADOR: {str(e)}\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)
        
    if hasattr(X_train_processed, 'columns'):
        all_cols = X_train_processed.columns.tolist()
    else:
        try:
            all_cols = list(preprocessor.get_feature_names_out())
        except:
            all_cols = [f"Feature_{i}" for i in range(X_train_processed.shape[1])]
            
    base_cat = config.get('categorical_features', [])
    base_bin = config.get('binary_features', [])
    
    features_categoricas = [col for col in all_cols if any(c in col for c in base_cat)]
    features_binarias = [col for col in all_cols if any(b in col for b in base_bin)]
    features_numericas = [col for col in all_cols if col not in features_categoricas and col not in features_binarias]
    
    cat_indices = [all_cols.index(col) for col in features_categoricas]
    
    log_and_track(f"Pré-processamento concluído | Total Features: {len(all_cols)}")
    
    groups_train = groups_train_val[tr_idx]
    groups_val = groups_train_val[val_idx]
    
    return {
        'preprocessor': preprocessor,
        'feature_names': all_cols,
        'cat_indices': cat_indices,
        'X_train_processed': X_train_processed, 'y_train': y_train,
        'X_val_processed': X_val_processed, 'y_val': y_val,
        'X_test_processed': X_test_processed, 'y_test': y_test,
        'X_train_val_processed': X_train_val_processed, 'y_train_val': y_train_val,
        'groups_train': groups_train, 'groups_val': groups_val,
        'groups_train_val': groups_train_val # <--- ADICIONAMOS A LISTA COMPLETA AQUI
    }

def search_hyperparameters_and_training_model(config, data_dict, tuner_func, get_focal_loss_obj_func, recalibrar_func, sample_base=True, search_method='grid', metodo=None, tracker=None):
    X_train_processed = data_dict['X_train_processed']
    y_train = data_dict['y_train']
    X_val_processed = data_dict['X_val_processed']
    y_val = data_dict['y_val']
    X_train_val_processed = data_dict['X_train_val_processed']
    y_train_val = data_dict['y_train_val']
    cat_indices = data_dict['cat_indices']
    groups_train = data_dict['groups_train']
    
    SAMPLE_SIZE = 1000000
    if sample_base and len(X_train_processed) > SAMPLE_SIZE:
        if tracker: tracker.update_node("step_3", "running", f"Gerando amostra estratificada...")
        unique_groups, first_indices = np.unique(groups_train, return_index=True)
        group_targets = y_train.values[first_indices] if hasattr(y_train, 'values') else y_train[first_indices]
        
        pct = SAMPLE_SIZE / len(X_train_processed)
        sampled_group_ids, _ = train_test_split(
            unique_groups, train_size=pct, stratify=group_targets, random_state=config.get('random_state', 42)
        )
        group_set = set(sampled_group_ids)
        mask = np.array([g in group_set for g in groups_train])
        
        X_train_sample = X_train_processed[mask] if isinstance(X_train_processed, np.ndarray) else X_train_processed.iloc[mask]
        y_train_sample = y_train[mask] if isinstance(y_train, np.ndarray) else y_train.iloc[mask]
    else:
        X_train_sample, y_train_sample = X_train_processed, y_train
        
        # ---> NOVO LOG DE QUANTIDADE DE LINHAS AQUI <---
        msg_amostra = f"📊 Base enviada para treino: {len(X_train_sample):,} linhas e {X_train_sample.shape[1]} variáveis."
        logger.info(msg_amostra)
        if tracker: tracker.update_node("step_3", "running", msg_amostra)
        
        if tracker: tracker.update_node("step_3", "running", "Iniciando Otimização de Hiperparâmetros...")
    melhores_params = tuner_func(
        X_train_sample, y_train_sample, X_val_processed, y_val, 
        cat_indices, config, get_focal_loss_obj_func, recalibrar_func, metodo=metodo, tracker=tracker
    )
    
    if melhores_params is None:
        raise ValueError("O otimizador falhou no Kill Switch de PD Alto. Tente usar o Grid Search ou abaixar o Scale Pos Weight.")

    final_params = melhores_params.copy()
    metodo_final = final_params.pop('metodo', metodo)
    
    w_train_real = final_params.get('scale_pos_weight', 1.0)
    if 'peso_efetivo' in final_params:
        w_train_real = final_params.pop('peso_efetivo')
        
    for key in ['logging_level', 'verbose', 'alpha', 'gamma', 'peso_efetivo', 'loss_function', 'eval_metric', 'iterations', 'w_train_calculado', 'metodo']:
        final_params.pop(key, None)
        
    loss_fn = 'Logloss'
    if metodo_final == 'focal':
        a_f = melhores_params.get('alpha', 0.25)
        g_f = melhores_params.get('gamma', 2.0)
        loss_fn = get_focal_loss_obj_func(alpha_val=a_f, gamma_val=g_f)
        final_params.pop('scale_pos_weight', None)
        
    # --- FÁBRICA DINÂMICA DE MODELOS ---
    algo = config.get('algorithm', 'catboost').lower()
    if tracker: tracker.update_node("step_3", "running", f"Instanciando modelo final: {algo.upper()}...")
    
    if algo == 'catboost':
        final_params.pop('max_bins', None)
        final_params.pop('interactions', None)
        final_params.pop('max_depth', None)
        model_final = cb.CatBoostClassifier(
            verbose=100, cat_features=cat_indices, early_stopping_rounds=50, loss_function=loss_fn, **final_params
        )
        model_final.fit(X_train_val_processed, y_train_val)
        
    elif algo == 'lightgbm':
        final_params.pop('depth', None)
        final_params.pop('max_bins', None)
        final_params.pop('interactions', None)
        final_params.pop('l2_leaf_reg', None)
        model_final = lgb.LGBMClassifier(random_state=42, verbose=-1, **final_params)
        model_final.fit(X_train_val_processed, y_train_val, categorical_feature=cat_indices)
        
    elif algo == 'ebm':
        final_params.pop('depth', None)
        final_params.pop('l2_leaf_reg', None)
        final_params.pop('max_depth', None)
        final_params.pop('scale_pos_weight', None) # EBM não aceita no construtor
        model_final = ExplainableBoostingClassifier(random_state=42, **final_params)
        # EBM usa sample_weight direto no fit
        sample_w = np.where(y_train_val == 1, w_train_real, 1.0)
        model_final.fit(X_train_val_processed, y_train_val, sample_weight=sample_w)
    else:
        raise ValueError(f"Algoritmo '{algo}' não suportado.")
    
    if tracker: tracker.update_node("step_3", "running", "Treinamento Final na Base Completa Concluído!")
    
    y_prob_raw = model_final.predict_proba(X_train_val_processed)[:, 1]
    y_prob_f = recalibrar_func(y_prob_raw, w_train_real)
    y_pred_f = (y_prob_f >= 0.5).astype(int)
    
    f1_f = f1_score(y_train_val, y_pred_f, zero_division=0)
    auc_f = roc_auc_score(y_train_val, y_prob_f)
    
    if tracker: tracker.update_node("step_3", "success", f"Treino concluído -> F1: {f1_f:.4f} | AUC: {auc_f:.4f}")
    
    melhores_params['w_train_calculado'] = w_train_real
    return model_final, melhores_params