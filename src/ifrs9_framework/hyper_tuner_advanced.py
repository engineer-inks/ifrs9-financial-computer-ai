import random
import gc
import numpy as np
import catboost as cb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, brier_score_loss
import logging

logger = logging.getLogger("IFRS9_Engine.Tuner")

def otimizar_hiperparametros_grid_search(X_train, y_train, X_val, y_val, cat_indices, config, get_focal_loss_obj_func, recalibrar_func, metodo='logloss'):
    """
    v89/v90: Grid Search Híbrido com suporte a Focal Loss e Logloss padrão.
    Integra calibração analítica (Shift) e fábrica de objetivos JIT.
    """
    logger.info("="*80)
    logger.info(f"--- INICIANDO GRID SEARCH [{metodo.upper()}] ---")
    
    # 1. CONFIGURAÇÃO DE ESPAÇO DE BUSCA FIXO
    depths = config.get('depths', [4, 6, 8])
    lrs = config.get('lrs', [0.03, 0.05])
    l2_regs = config.get('l2_regs', [30, 100])
    
    F1_TARGET = config.get('f1_target', 0.33)
    PD_MAX_LIMIT = config.get('pd_kill_switch', 0.10)
    
    # 2. CONFIGURAÇÃO ESPECÍFICA POR MÉTODO
    if metodo == 'focal':
        alphas = config.get('alphas', [0.25, 0.35, 0.50])
        gammas = config.get('gammas', [2.0, 3.0, 5.0])
        grid_params = [(a, g, d, lr, l2) for a in alphas for g in gammas for d in depths for lr in lrs for l2 in l2_regs]
    else:
        # Lógica de Pesos para Logloss
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_arr), y=y_train_arr)
        base_weight = weights[1]
        fatores = config.get('scale_factors', [0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
        peso_sqrt = np.sqrt(base_weight)
        
        pesos_grid = sorted(list(set([base_weight * f for f in fatores] + [peso_sqrt])))
        logger.info(f"⚖️ Modo Logloss | Peso Base: {base_weight:.2f} | Peso Sqrt: {peso_sqrt:.2f}")
        
        grid_params = [(w, None, d, lr, l2) for w in pesos_grid for d in depths for lr in lrs for l2 in l2_regs]

    best_combined_score = -float('inf')
    melhores_params = None
    total_comb = len(grid_params)
    count = 0
    
    # 3. LOOP UNIFICADO
    for p1, p2, d, lr, l2 in grid_params:
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
            
        params = {
            'depth': d, 'learning_rate': lr, 'l2_leaf_reg': l2,
            'iterations': 1000, 'eval_metric': 'F1',
            'random_state': config.get('random_state', 42),
            'logging_level': 'Silent', 'thread_count': -1,
            'loss_function': loss_fn,
            'scale_pos_weight': scale_pos_weight
        }
        
        try:
            model = cb.CatBoostClassifier(**params, cat_features=cat_indices)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
            
            # --- RECALIBRAÇÃO E MÉTRICAS ---
            y_prob_raw = model.predict_proba(X_val, ntree_end=model.best_iteration_)[:, 1]
            y_prob_calib = recalibrar_func(y_prob_raw, final_weight)
            
            val_brier = brier_score_loss(y_val, y_prob_calib)
            y_pred = (y_prob_calib >= 0.5).astype(int)
            val_f1 = f1_score(y_val, y_pred, zero_division=0)
            pd_medio = np.mean(y_prob_calib)
            
            # 🔴 TRAVA DE SEGURANÇA (KILL SWITCH)
            if pd_medio > PD_MAX_LIMIT:
                logger.warning(f"⚠️ Kill Switch: PD de {pd_medio:.2%} é irreal. Pulando iteração...")
                continue
                
            combined_score = val_f1 - (val_brier * 2)
            
            if count % 20 == 0 or combined_score > best_combined_score:
                logger.info(f"[{count}/{total_comb}] W_Efetivo:{final_weight:.2f} | F1:{val_f1:.4f} | Brier:{val_brier:.4f}")
                
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                melhores_params = params.copy()
                melhores_params.update({
                    'iterations': model.best_iteration_,
                    'w_train_calculado': final_weight,
                    'peso_efetivo': final_weight,
                    'metodo': metodo
                })
                
                if metodo == 'focal':
                    melhores_params.update({'alpha': alpha, 'gamma': gamma})
                    
                logger.info(f"   🌟 TOP: F1 {val_f1:.4f} | Brier: {val_brier:.5f} | PD: {pd_medio:.4%}")
                
                # META ATINGIDA: Encerramento antecipado
                if val_f1 >= F1_TARGET and val_brier < 0.02:
                    logger.info("🎯 Meta atingida. Encerrando o Grid Search.")
                    return melhores_params

        except Exception as e:
            logger.error(f"❌ Erro na iteração {count}: {str(e)}")
            continue
        finally:
            if 'model' in locals(): del model
            gc.collect()
            
    logger.info("="*80)
    return melhores_params