import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import logging
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit

# Cores Padrão para os relatórios
PRETO = "#000000"
AZUL_PRIMARIO = "#1526FF"
CINZA_CLARO = "#CCCCCC"
CINZA_ESCURO = "#333333"

logger = logging.getLogger("IFRS9_Engine.Advanced")

# ==========================================
# 1. DIVISÃO SEGURA DE DADOS (ANTI-LEAKAGE)
# ==========================================
def div_train_test_split(df, config):
    """
    1. DIVISÃO COM DATAS PRESERVADAS (USANDO GROUP SPLIT)
    Garante que o mesmo grupo (ex: COD_OPR_ATV) não apareça no treino e teste.
    """
    target = config.get('target_column', 'default_flag')
    group_col = config.get('group_column', 'COD_OPR_ATV')
    
    # Validação de segurança
    if group_col not in df.columns:
        raise ValueError(f"Coluna de agrupamento '{group_col}' não encontrada!")
        
    logger.info(f"Realizando Split Treino/Teste por Grupos de contrato ({group_col})...")
    
    # Split Treino+Val vs Teste
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.get('test_size', 0.25), random_state=config.get('random_state', 42))
    train_val_idx, test_idx = next(splitter.split(df, df[target], groups=df[group_col]))
    
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    logger.info(f"Shape Treino+Val: {df_train_val.shape}, Shape Teste: {df_test.shape}")
    
    return df_train_val, df_test

# ==========================================
# 2. OTIMIZADOR IFRS 9 (F1 vs RATIO)
# ==========================================
def otimizar_pesos_e_threshold_f1(y_true, y_prob_raw, output_dir, w_train=1.0, recall_minimo=0.20):
    """
    v78: Maximiza F1-Score garantindo que o Ratio (PD Predita / PD Real)
    esteja dentro dos limites regulatórios do IFRS9 (0.7 a 1.3).
    """
    logger.info("--- INICIANDO BUSCA EM GRADE COM TRAVA DE RATIO (v78) ---")
    
    # 1. NEUTRALIZAÇÃO INICIAL (Obrigatório para evitar Ratio de 20x)
    y_prob_neutral = y_prob_raw / (y_prob_raw + (w_train * (1 - y_prob_raw)))
    pd_real = y_true.mean()
    
    # Definição da Grade de Refinamento
    pesos_teste = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    thresholds_teste = np.linspace(0.05, 0.50, 20)
    
    results = []
    best_f1 = -1
    best_config = None
    
    for w in pesos_teste:
        # Aplica Shift de Calibração
        y_prob_shifted = y_prob_neutral / (y_prob_neutral + (1 - y_prob_neutral) / w)
        
        for t in thresholds_teste:
            y_pred = (y_prob_shifted >= t).astype(int)
            
            # Cálculo de Métricas
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Cálculo do Ratio (A métrica de segurança do IFRS9)
            pd_predita = y_pred.mean()
            ratio = pd_predita / pd_real if pd_real > 0 else 0
            
            # PENALIZAÇÃO DE RATIO: Reduzimos o F1 drasticamente se ratio for absurdo
            f1_penalizado = f1 if (0.7 <= ratio <= 1.3) else (f1 * 0.1)
            
            results.append({
                'peso': w, 'threshold': t, 'f1': f1,
                'recall': rec, 'precision': prec, 'ratio': ratio,
                'f1_penalizado': f1_penalizado
            })
            
            # Critério de Seleção
            if rec >= recall_minimo and f1_penalizado > best_f1:
                best_f1 = f1_penalizado
                best_config = {
                    'peso': w, 'threshold': t, 'f1': f1,
                    'recall': rec, 'prec': prec, 'ratio': ratio
                }
                
    # Se a trava foi muito forte, pegamos o melhor absoluto (Relaxando restrições)
    if best_config is None:
        logger.warning("⚠️ Nenhuma configuração atendeu à trava de Ratio 0.7-1.3. Relaxando restrições...")
        df_res = pd.DataFrame(results)
        best_config = df_res.loc[df_res['f1'].idxmax()].to_dict()
        
    # --- GERAR HEATMAP (Padrão de Produção) ---
    os.makedirs(output_dir, exist_ok=True)
    df_plot = pd.DataFrame(results)
    pivot_f1 = df_plot.pivot(index='peso', columns='threshold', values='f1')
    
    plt.figure(figsize=(14, 9), facecolor=PRETO)
    sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="magma", cbar_kws={'label': 'F1-Score'})
    
    plt.title(f'Otimização F1 com Trava de Ratio\nMelhor F1: {best_config["f1"]:.2%} | Ratio: {best_config["ratio"]:.2f}x', 
              color='white', fontsize=14, fontweight='bold', pad=20)
              
    save_path = os.path.join(output_dir, 'grid_search_f1_calibrado_v78.png')
    plt.savefig(save_path, dpi=130, facecolor=PRETO, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ CONFIGURAÇÃO OTIMIZADA: Peso={best_config['peso']} | T={best_config['threshold']:.2f}")
    logger.info(f"📊 PERFORMANCE: F1={best_config['f1']:.2f} | Ratio={best_config['ratio']:.2f}x | Recall={best_config['recall']:.2f}")
    
    return best_config

# ==========================================
# 3. MÉTRICAS E MATRIZ DE CONFUSÃO IFRS 9
# ==========================================
def gerar_metricas_classificacao(y_true, y_prob, output_dir, threshold, method='modelo'):
    """
    v92: Versão Final Consolidada.
    Gera métricas de classificação, Matriz de Confusão e logs de auditoria IFRS9.
    """
    # 1. TRATAMENTO DO THRESHOLD
    t_corte = threshold['threshold'] if isinstance(threshold, dict) else threshold
    logger.info(f"--- Gerando Métricas de Classificação (Threshold: {t_corte:.4f}) ---")
    
    # 2. BINARIZAÇÃO E MATRIZ
    y_pred = (y_prob >= t_corte).astype(int)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # 3. CÁLCULO DE MÉTRICAS
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Razão de Predição
    ratio_classif = y_pred.mean() / (y_true.mean() + 1e-9)
    
    # 4. LOGS ESTRUTURADOS (Padrão IFRS9 exato do print)
    logger.info("\n" + "="*80)
    logger.info(f"MATRIZ DE CONFUSÃO - MÉTODO: {method.upper()}")
    logger.info(f"PONTO DE CORTE OTIMIZADO: {t_corte:.2%}")
    logger.info("-" * 80)
    logger.info(f"BONS Reais (0): {tn+fp:<10,} | Acertos: {tn:<10,} (TN) | Erros: {fp:<10,} (FP)")
    logger.info(f"MAUS Reais (1): {fn+tp:<10,} | Acertos: {tp:<10,} (TP) | Erros: {fn:<10,} (FN)")
    logger.info("-" * 80)
    logger.info(f"📊 PERFORMANCE: ACC: {acc:.4f} | MCC: {mcc:.4f} | Ratio: {ratio_classif:.2f}x")
    logger.info(f"📈 DECISÃO:     PRECISION: {precision:.4f} | RECALL: {recall:.4f} | F1: {f1:.4f}")
    logger.info("="*80)
    
    # 5. PLOTAGEM EXECUTIVA (Padrão v29)
    plt.figure(figsize=(10, 8), facecolor=PRETO)
    ax = plt.gca()
    ax.set_facecolor(PRETO)
    
    # Colormap: Preto -> Azul Primário
    cmap_banco = mcolors.LinearSegmentedColormap.from_list("banco_azul", [PRETO, AZUL_PRIMARIO])
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap_banco,
                annot_kws={'size': 16, 'fontweight': 'bold', 'color': 'white'},
                xticklabels=['Previsto BOM', 'Previsto MAU'],
                yticklabels=['Real BOM', 'Real MAU'],
                cbar=False, ax=ax)
                
    plt.title(f'Matriz de Confusão - {method.upper()}\n(Threshold: {t_corte:.2%})',
              color='white', fontsize=16, fontweight='bold', pad=25)
              
    plt.ylabel('Verdadeiro (Real)', color=CINZA_CLARO, fontsize=12)
    plt.xlabel('Predito (Modelo)', color=CINZA_CLARO, fontsize=12)
    ax.tick_params(axis='both', colors=CINZA_CLARO, labelsize=11)
    
    # Moldura Estilizada
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color(CINZA_ESCURO)
        
    plt.tight_layout()
    
    # Salvamento Dinâmico
    os.makedirs(output_dir, exist_ok=True)
    filename = f'matriz_confusao_{method.lower()}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=130, facecolor=PRETO)
    plt.close()
    
    return {
        'metodo': method, 'precision': precision, 'recall': recall, 'f1_score': f1,
        'ratio_classif': ratio_classif, 'threshold_usado': t_corte, 'acc': acc, 'mcc': mcc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    
def salvar_metricas_csv(config, metrics_dict, nome_arquivo="relatorio_performance.csv"):
    """Salva um dicionário de métricas em CSV"""
    df = pd.DataFrame([metrics_dict])
    out_dir = config.get('metrics', config.get('GRAPHICS_DIR', 'outputs'))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, nome_arquivo)
    df.to_csv(path, index=False, sep=';')
    logger.info(f"✅ Relatório CSV salvo: {path}")