import numpy as np
import pandas as pd
import logging
import catboost as cb
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss

logger = logging.getLogger("IFRS9_Engine.Training")

# ==========================================
# 1. PREPARAÇÃO DAS MATRIZES E FEATURES
# ==========================================
def preparar_matrizes(df_train_val, df_test, config, preprocessor_func):
    """
    Separa X e Y, remove colunas constantes e aplica o pré-processamento.
    Garante que não há data leakage (remoção baseada apenas no treino).
    """
    # Lendo dinamicamente da config em vez de usar hardcode
    target = config.get('target_column', 'default_flag')
    group_col = config.get('group_column', 'COD_OPR_ATV')
    cols_to_drop_config = config.get('columns_to_drop', [])
    
    def split_X_y_raw(df_subset):
        y_out = df_subset[target]
        cols_to_drop = [target] + [c for c in cols_to_drop_config if c in df_subset.columns]
        X_out = df_subset.drop(columns=cols_to_drop)
        return X_out, y_out
        
    # 1. Separa X e y Brutos
    X_train_val_raw, y_train_val = split_X_y_raw(df_train_val)
    X_test_raw, y_test = split_X_y_raw(df_test)
    
    # 2. Identifica colunas constantes APENAS NO TREINO (Blindagem contra Leakage)
    cols_constantes = [col for col in X_train_val_raw.columns if X_train_val_raw[col].nunique(dropna=False) <= 1]
    if cols_constantes:
        logger.info(f"Removendo {len(cols_constantes)} colunas constantes encontradas no treino.")
        X_train_val_raw.drop(columns=cols_constantes, inplace=True)
        # Aplica a MESMA remoção no teste (garante alinhamento exato das matrizes)
        X_test_raw.drop(columns=cols_constantes, inplace=True, errors='ignore')
        
    logger.debug(f'Features de treinamento enviadas: {list(X_train_val_raw.columns)}')
    
    # 3. Segundo Split: Treino vs Validação (Interno para Early Stopping)
    groups_train_val = df_train_val[group_col].values
    logger.info(f"Realizando Split Treino/Validação por Grupos ({group_col})...")
    
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.get('random_state', 42))
    tr_idx, val_idx = next(splitter_val.split(X_train_val_raw, y_train_val, groups=groups_train_val))
    
    X_train_raw = X_train_val_raw.iloc[tr_idx]
    y_train = y_train_val.iloc[tr_idx]
    X_val_raw = X_train_val_raw.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]
    
    logger.info(f"Linhas finais -> Treino: {X_train_raw.shape[0]}, Validação: {X_val_raw.shape[0]}, Teste: {X_test_raw.shape[0]}")
    
    # 4. Aplicação do Pré-Processador (O Motor Scikit-Learn que criámos noutro ficheiro)
    preprocessor = preprocessor_func(X_train_raw, config, logger)
    logger.info("Aplicando pré-processamento...")
    
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_val_processed = preprocessor.transform(X_val_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    X_train_val_processed = preprocessor.transform(X_train_val_raw)
    
    # 5. Identificação Robusta de Categóricas e Nomes pós-transformação
    if hasattr(X_train_processed, 'columns'):
        all_cols = X_train_processed.columns.tolist()
    else:
        # Caso o sklearn devolva NumPy (comum se sparse=False), pegamos nomes do preprocessor
        try:
            all_cols = list(preprocessor.get_feature_names_out())
        except Exception:
            all_cols = [f"Feature_{i}" for i in range(X_train_processed.shape[1])]
            
    base_cat = config.get('categorical_features', [])
    base_bin = config.get('binary_features', [])
    
    features_categoricas = [col for col in all_cols if any(c in col for c in base_cat)]
    features_binarias = [col for col in all_cols if any(b in col for b in base_bin)]
    features_numericas = [col for col in all_cols if col not in features_categoricas and col not in features_binarias]
    
    # Índices exatos para alimentar o CatBoost sem erros
    cat_indices = [all_cols.index(col) for col in features_categoricas]
    n_num_final = len(features_numericas)
    
    logger.info("\n" + "="*60)
    logger.info(" EXAME FINAL DE FEATURES - INPUT DO MODELO")
    logger.info("="*60)
    logger.info(f" TOTAL DE FEATURES: {len(all_cols)}")
    logger.info(f" NUMÉRICAS: {len(features_numericas)}")
    logger.info(f" CATEGÓRICAS: {len(features_categoricas)}")
    logger.info(f" BINÁRIAS: {len(features_binarias)}")
    logger.info(f" ÍNDICES CATEGÓRICOS: {cat_indices}")
    logger.info("="*60 + "\n")
    
    groups_train = groups_train_val[tr_idx]
    groups_val = groups_train_val[val_idx]
    logger.info(f"Grupos separados -> Treino: {len(np.unique(groups_train))}, Validação: {len(np.unique(groups_val))}")
    
    return {
        'preprocessor': preprocessor,
        'feature_names': all_cols,
        'cat_indices': cat_indices,
        'X_train_processed': X_train_processed, 'y_train': y_train,
        'X_val_processed': X_val_processed, 'y_val': y_val,
        'X_test_processed': X_test_processed, 'y_test': y_test,
        'X_train_val_processed': X_train_val_processed, 'y_train_val': y_train_val,
        'groups_train': groups_train, 'groups_val': groups_val
    }

# ==========================================
# 2. MOTOR HÍBRIDO DE TREINO E BUSCA
# ==========================================
def search_hyperparameters_and_training_model(config, data_dict, tuner_func, get_focal_loss_obj_func, recalibrar_func, sample_base=True, search_method='grid', metodo=None):
    """
    v91: Treinamento Final Híbrido com Calibração de Saída e Limpeza de Metadados.
    """
    # Extração elegante do dicionário
    X_train_processed = data_dict['X_train_processed']
    y_train = data_dict['y_train']
    X_val_processed = data_dict['X_val_processed']
    y_val = data_dict['y_val']
    X_train_val_processed = data_dict['X_train_val_processed']
    y_train_val = data_dict['y_train_val']
    cat_indices = data_dict['cat_indices']
    groups_train = data_dict['groups_train']
    
    # 1. AMOSTRAGEM CENTRALIZADA (Evita rebentar a RAM em bases massivas > 1M)
    SAMPLE_SIZE = 1000000
    if sample_base and len(X_train_processed) > SAMPLE_SIZE:
        logger.info(f"⚡ Base muito grande. Gerando amostra estratificada de {SAMPLE_SIZE} linhas para Tuning rápido...")
        
        # Garante que não quebramos os grupos ao fazer sample!
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
        
    # 2. BUSCA DE HIPERPARÂMETROS (Invoca o Grid Search que criámos)
    logger.info("🔍 Iniciando Busca de Hiperparâmetros no Motor Analítico...")
    melhores_params = tuner_func(
        X_train_sample, y_train_sample, X_val_processed, y_val, 
        cat_indices, config, get_focal_loss_obj_func, recalibrar_func, metodo=metodo
    )
    
    if melhores_params is None:
        raise ValueError("O otimizador não conseguiu encontrar nenhum conjunto de parâmetros que satisfaça a trava do IFRS9 (PD média aceitável).")

    # 3. CONFIGURAÇÃO FINAL E LIMPEZA
    final_params = melhores_params.copy()
    metodo_final = final_params.pop('metodo', metodo)
    
    # Extração de pesos efetivos para report futuro
    w_train_real = final_params.get('scale_pos_weight', 1.0)
    if 'peso_efetivo' in final_params:
        w_train_real = final_params.pop('peso_efetivo')
        
    # Limpeza de chaves de auditoria que o CatBoost não aceita no construtor
    for key in ['logging_level', 'verbose', 'alpha', 'gamma', 'peso_efetivo']:
        final_params.pop(key, None)
        
    # Re-instanciação do Objetivo se for Focal
    loss_fn = 'Logloss'
    if metodo_final == 'focal':
        a_f = melhores_params.get('alpha', 0.25)
        g_f = melhores_params.get('gamma', 2.0)
        loss_fn = get_focal_loss_obj_func(alpha_val=a_f, gamma_val=g_f)
        # O CatBoost não aceita scale_pos_weight junto com custom_loss
        final_params.pop('scale_pos_weight', None)
        
    logger.info(f"⚙️ Configuração Final do Modelo: {final_params}")
    
    model_final = cb.CatBoostClassifier(
        verbose=100,
        cat_features=cat_indices,
        early_stopping_rounds=50,
        loss_function=loss_fn,
        **final_params
    )
    
    logger.info("--- Treinando Modelo Final Vencedor na Base Completa (Train + Val) ---")
    model_final.fit(X_train_val_processed, y_train_val)
    
    # 4. RELATÓRIO DE PERFORMANCE CALIBRADO (v91)
    y_prob_raw = model_final.predict_proba(X_train_val_processed)[:, 1]
    
    # Aqui aplicamos o Shift para que o Log reflita a PD real do banco e não o viés do peso
    y_prob_f = recalibrar_func(y_prob_raw, w_train_real)
    y_pred_f = (y_prob_f >= 0.5).astype(int)
    
    f1_f = f1_score(y_train_val, y_pred_f, zero_division=0)
    prec_f = precision_score(y_train_val, y_pred_f, zero_division=0)
    rec_f = recall_score(y_train_val, y_pred_f, zero_division=0)
    auc_f = roc_auc_score(y_train_val, y_prob_f)
    brier_f = brier_score_loss(y_train_val, y_prob_f)
    pd_medio = np.mean(y_prob_f)
    
    logger.info("="*80)
    logger.info(f"📊 PERFORMANCE FINAL CALIBRADA NA BASE TREINO (Peso Shift: {w_train_real:.2f})")
    logger.info(f"   🎯 F1-Score  : {f1_f:.4f}")
    logger.info(f"   🎯 Precision : {prec_f:.2%}  | 📈 Recall: {rec_f:.2%}")
    logger.info(f"   🔍 AUC       : {auc_f:.4f}  | ⚖️ Brier : {brier_f:.6f}")
    logger.info(f"   💰 PD Média  : {pd_medio:.4%}")
    logger.info("="*80)
    
    # Anexamos o peso real encontrado de volta para facilitar o OOF
    melhores_params['w_train_calculado'] = w_train_real
    
    return model_final, melhores_params