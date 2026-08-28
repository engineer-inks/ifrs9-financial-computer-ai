import random
import gc
import numpy as np
import catboost as cb
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, brier_score_loss
import logging

logger = logging.getLogger("IFRS9_Engine.Tuner")

def otimizar_hiperparametros_grid_search(X_train, y_train, X_val, y_val, cat_indices, config, get_focal_loss_obj_func, recalibrar_func, metodo='logloss', tracker=None):
    algo = config.get('algorithm', 'catboost').lower()
    msg_init = f"--- INICIANDO GRID SEARCH [{metodo.upper()}] PARA {algo.upper()} ---"
    logger.info("="*80)
    logger.info(msg_init)
    if tracker: tracker.update_node("step_3", "running", msg_init)
    
    # Lendo os hiperparâmetros diretamente da Interface Gráfica (UI)
    hp = config.get('hyperparameters', {})
    
    # 1. MAPEAMENTO DINÂMICO DOS PARÂMETROS
    if algo == 'ebm':
        interactions = hp.get('interactions', [0, 10])
        if not isinstance(interactions, list): interactions = [interactions]
        max_bins = hp.get('max_bins', [64, 256])
        if not isinstance(max_bins, list): max_bins = [max_bins]
        lrs = hp.get('learning_rate', [0.01, 0.05])
        if not isinstance(lrs, list): lrs = [lrs]
        grid_params_base = [(mb, inter, lr) for mb in max_bins for inter in interactions for lr in lrs]
        
    elif algo == 'lightgbm':
        depths = hp.get('max_depth', [4, 6])
        if not isinstance(depths, list): depths = [depths]
        leaves = hp.get('num_leaves', [31, 50])
        if not isinstance(leaves, list): leaves = [leaves]
        lrs = hp.get('learning_rate', [0.03, 0.05])
        if not isinstance(lrs, list): lrs = [lrs]
        grid_params_base = [(d, l, lr) for d in depths for l in leaves for lr in lrs]
        
    else: # catboost
        depths = hp.get('depth', [4, 6])
        if not isinstance(depths, list): depths = [depths]
        lrs = hp.get('learning_rate', [0.03, 0.05])
        if not isinstance(lrs, list): lrs = [lrs]
        l2_regs = hp.get('l2_leaf_reg', [3, 10])
        if not isinstance(l2_regs, list): l2_regs = [l2_regs]
        grid_params_base = [(d, l2, lr) for d in depths for l2 in l2_regs for lr in lrs]

    F1_TARGET = config.get('f1_target', 0.33)
    PD_MAX_LIMIT = config.get('pd_kill_switch', 0.10)
    
    # 2. CONFIGURAÇÃO DOS PESOS E LOSS FUNCTION
    if metodo == 'focal':
        alphas = config.get('alphas', [0.25, 0.50])
        gammas = config.get('gammas', [2.0, 3.0])
        grid_params = [(a, g, p[0], p[1], p[2]) for a in alphas for g in gammas for p in grid_params_base]
    else:
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_arr), y=y_train_arr)
        base_weight = weights[1]
        
        ui_weights = hp.get('scale_pos_weight', [base_weight * 0.5, base_weight])
        if not isinstance(ui_weights, list): ui_weights = [ui_weights]
        
        pesos_grid = sorted(list(set(ui_weights + [base_weight])))
        
        msg_logloss = f"⚖️ Modo Logloss | Pesos base a testar: {[round(w, 2) for w in pesos_grid]}"
        logger.info(msg_logloss)
        if tracker: tracker.update_node("step_3", "running", msg_logloss)
        
        grid_params = [(w, None, p[0], p[1], p[2]) for w in pesos_grid for p in grid_params_base]

    best_combined_score = -float('inf')
    melhores_params = None
    total_comb = len(grid_params)
    count = 0
    
    # 3. LOOP UNIFICADO DE TREINAMENTO
    for p1, p2, param_a, param_b, param_c in grid_params:
        count += 1
        
        if metodo == 'focal':
            alpha, gamma = p1, p2
            final_weight = alpha / (1 - alpha)
            loss_fn = get_focal_loss_obj_func(alpha_val=alpha, gamma_val=gamma)
            scale_pos_weight = None
        else:
            final_weight = p1
            loss_fn = 'Logloss'
            scale_pos_weight = final_weight
            
        params = {'random_state': config.get('random_state', 42)}
        
        # LOG IMEDIATO ANTES DO FIT PARA A UI NÃO PARECER CONGELADA
        msg_iter = f"⏳ [{count}/{total_comb}] Iniciando treino... (Peso: {final_weight:.2f})"
        logger.info(msg_iter)
        if tracker: tracker.update_node("step_3", "running", msg_iter)
        
        try:
            if algo == 'catboost':
                d, l2, lr = param_a, param_b, param_c
                params.update({'depth': int(d), 'l2_leaf_reg': l2, 'learning_rate': lr, 'iterations': 100, 'eval_metric': 'F1', 'logging_level': 'Silent', 'thread_count': -1, 'loss_function': loss_fn, 'scale_pos_weight': scale_pos_weight})
                model = cb.CatBoostClassifier(**params, cat_features=cat_indices)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
                best_iter = model.best_iteration_
                y_prob_raw = model.predict_proba(X_val, ntree_end=best_iter)[:, 1]
                
            elif algo == 'lightgbm':
                d, leaves, lr = param_a, param_b, param_c
                params.update({'max_depth': int(d), 'num_leaves': int(leaves), 'learning_rate': lr, 'n_estimators': 100, 'scale_pos_weight': scale_pos_weight, 'verbose': -1})
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], categorical_feature=cat_indices)
                best_iter = model.best_iteration_
                y_prob_raw = model.predict_proba(X_val)[:, 1]
                
            elif algo == 'ebm':
                mb, inter, lr = param_a, param_b, param_c
                params.update({'max_bins': int(mb), 'interactions': int(inter), 'learning_rate': lr})
                
                # AVISO DE LENTIDÃO PARA O EBM NA UI
                if tracker: tracker.update_node("step_3", "running", f"⏳ [{count}/{total_comb}] EBM calculando {int(inter)} interações emparelhadas. Isto pode demorar vários minutos...")
                
                model = ExplainableBoostingClassifier(**params)
                sample_w = np.where(y_train == 1, final_weight, 1.0)
                model.fit(X_train, y_train, sample_weight=sample_w)
                best_iter = 100
                y_prob_raw = model.predict_proba(X_val)[:, 1]
            
            # --- RECALIBRAÇÃO E AVALIAÇÃO DO CICLO ---
            y_prob_calib = recalibrar_func(y_prob_raw, final_weight)
            val_brier = brier_score_loss(y_val, y_prob_calib)
            y_pred = (y_prob_calib >= 0.5).astype(int)
            val_f1 = f1_score(y_val, y_pred, zero_division=0)
            pd_medio = np.mean(y_prob_calib)
            
            if pd_medio > PD_MAX_LIMIT:
                msg_kill = f"⚠️ Kill Switch: PD de {pd_medio:.2%} é alto. "
                if total_comb > 1:
                    msg_kill += "Pulando iteração..."
                    logger.warning(msg_kill)
                    if tracker: tracker.update_node("step_3", "running", msg_kill)
                    continue
                else:
                    msg_kill += "Ignorado por ser Modo Manual."
                    logger.warning(msg_kill)
                    if tracker: tracker.update_node("step_3", "running", msg_kill)
                
            combined_score = val_f1 - (val_brier * 2)
            
            msg_result = f"✅ [{count}/{total_comb}] Concluído -> F1: {val_f1:.4f} | Brier: {val_brier:.4f}"
            logger.info(msg_result)
            if tracker: tracker.update_node("step_3", "running", msg_result)
                
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                melhores_params = params.copy()
                melhores_params.update({
                    'iterations': best_iter,
                    'w_train_calculado': final_weight,
                    'peso_efetivo': final_weight,
                    'metodo': metodo
                })
                
                if metodo == 'focal':
                    melhores_params.update({'alpha': alpha, 'gamma': gamma})
                    
                msg_top = f"   🌟 NOVO TOP: F1 {val_f1:.4f} | Brier: {val_brier:.5f} | PD: {pd_medio:.4%}"
                logger.info(msg_top)
                if tracker: tracker.update_node("step_3", "running", msg_top)
                
                # META ATINGIDA: Encerramento antecipado
                if val_f1 >= F1_TARGET and val_brier < 0.02:
                    msg_meta = "🎯 Meta atingida. Encerrando o Grid Search."
                    logger.info(msg_meta)
                    if tracker: tracker.update_node("step_3", "running", msg_meta)
                    return melhores_params

        except Exception as e:
            msg_err = f"❌ Erro na iteração {count}: {str(e)}"
            logger.error(msg_err)
            if tracker: tracker.update_node("step_3", "running", msg_err)
            continue
        finally:
            if 'model' in locals(): del model
            gc.collect()
            
    logger.info("="*80)
    return melhores_params