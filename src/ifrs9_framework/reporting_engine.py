import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import logging

from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, recall_score, precision_score
from scipy.stats import ks_2samp

logger = logging.getLogger("IFRS9_Engine.Reporting")

PRETO = "#000000"
AZUL_PRIMARIO = "#1526FF"
AZUL_SECUNDARIO = "#0066FC"
ROSA = "#FF007F"
CINZA_CLARO = "#CCCCCC"
CINZA_ESCURO = "#333333"

def gerar_analise_temporal(df_test_completo: pd.DataFrame, y_true: pd.Series, y_prob: np.ndarray, 
                           output_dir: str, col_data: str, label_periodo: str, filename_suffix: str):
    """Gera gráfico de Backtesting empilhado (Volume em cima, Taxas em baixo)."""
    if col_data not in df_test_completo.columns:
        logger.warning(f"⚠️ Coluna '{col_data}' não encontrada.")
        return
        
    df_analise = df_test_completo[[col_data]].copy().reset_index(drop=True)
    df_analise['Default_Real'] = y_true.values if hasattr(y_true, 'values') else y_true
    df_analise['PD_Modelo'] = y_prob
    
    df_analise['DATA_REF'] = pd.to_datetime(df_analise[col_data], errors='coerce')
    df_analise.dropna(subset=['DATA_REF'], inplace=True)
    df_analise['periodo'] = df_analise['DATA_REF'].dt.to_period('M')
    
    temporal_df = df_analise.groupby('periodo', observed=False).agg(
        Vol=('PD_Modelo', 'count'),
        PD_Media=('PD_Modelo', 'mean'),
        Default_Rate=('Default_Real', 'mean')
    ).reset_index()
    
    temporal_df = temporal_df[temporal_df['Vol'] > 50].sort_values('periodo')
    temporal_df['periodo_ts'] = temporal_df['periodo'].dt.to_timestamp()
    
    fig, (ax_vol, ax_line) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                          gridspec_kw={'height_ratios': [1, 3]},
                                          facecolor=PRETO)
                                          
    fig.subplots_adjust(hspace=0.05)
    ax_vol.set_facecolor(PRETO)
    ax_line.set_facecolor(PRETO)
    
    ax_vol.bar(temporal_df['periodo_ts'], temporal_df['Vol'], 
               color=AZUL_SECUNDARIO, alpha=0.4, width=20, label='Volume de Contratos')
               
    ax_vol.set_ylabel('Volume', color=CINZA_CLARO, fontsize=10)
    ax_vol.tick_params(axis='y', labelcolor=CINZA_ESCURO)
    ax_vol.grid(True, alpha=0.1, color=CINZA_CLARO)
    plt.setp(ax_vol.get_xticklabels(), visible=False)
    ax_vol.tick_params(axis='x', which='both', bottom=False, top=False)
    
    ax_line.plot(temporal_df['periodo_ts'], temporal_df['PD_Media'],
                 color=AZUL_PRIMARIO, linewidth=4, marker='o', markersize=7, label='PD Modelo (Esperado)')
                 
    ax_line.plot(temporal_df['periodo_ts'], temporal_df['Default_Rate'],
                 color=ROSA, linewidth=3, marker='X', markersize=8, linestyle='--', label='Default Real (Observado)')
                 
    ax_line.set_ylabel('Taxa de Inadimplência / PD', color=CINZA_CLARO, fontsize=12)
    ax_line.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
    ax_line.grid(True, alpha=0.15, color=CINZA_CLARO)
    
    ax_line.set_xlabel('Safra / Período de Referência', color=CINZA_CLARO, fontsize=12)
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    intervalo = 3 if len(temporal_df) > 18 else 1
    ax_line.xaxis.set_major_locator(mdates.MonthLocator(interval=intervalo))
    plt.setp(ax_line.get_xticklabels(), rotation=45, ha='right', color=CINZA_CLARO)
    
    fig.suptitle(f'Backtesting de Risco: {label_periodo}\nAnálise de Estabilidade e Aderência',
                 color='white', fontsize=18, fontweight='bold', y=0.96)
                 
    ax_vol.legend(loc='upper left', facecolor=PRETO, edgecolor=CINZA_ESCURO, fontsize=9)
    ax_line.legend(loc='upper left', facecolor=PRETO, edgecolor=CINZA_ESCURO, fontsize=11)
    
    for ax in [ax_vol, ax_line]:
        ax.tick_params(colors=CINZA_CLARO)
        for spine in ax.spines.values():
            spine.set_edgecolor(CINZA_ESCURO)
            
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'backtesting_stacked_{filename_suffix}.png')
    plt.savefig(save_path, dpi=130, facecolor=PRETO, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Backtesting Empilhado salvo em: {save_path}")

def load_depara_features(feat_names_tecnicos, config):
    depara = config.get('depara_features', {})
    return [depara.get(f, f) for f in feat_names_tecnicos]

def plotar_feature_importance(model_final, feature_names, X_test, config, plot_dir):
    try:
        feature_importance = model_final.get_feature_importance()
    except AttributeError:
        feature_importance = model_final.feature_importances_
        
    if feature_names is not None:
        feat_names_tecnicos = feature_names
    elif hasattr(X_test, 'columns'):
        feat_names_tecnicos = X_test.columns.tolist()
    else:
        feat_names_tecnicos = [f"Feature_{i}" for i in range(len(feature_importance))]
        
    feat_names_amigaveis = load_depara_features(feat_names_tecnicos, config)
    
    df_imp = pd.DataFrame({
        'feature': feat_names_amigaveis,
        'feature_tecnica': feat_names_tecnicos,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(25)
    
    plt.figure(figsize=(14, 10), facecolor=PRETO)
    ax = plt.gca()
    ax.set_facecolor(PRETO)
    
    colors = [ROSA if i == 0 else AZUL_PRIMARIO for i in range(len(df_imp))]
    sns.barplot(data=df_imp, x='importance', y='feature', palette=colors, ax=ax)
    
    plt.title('Top 25 Drivers de Decisão do Modelo', color='white', fontsize=18, fontweight='bold', pad=30)
    plt.xlabel('Importância Relativa (Score de Contribuição)', color=CINZA_CLARO, fontsize=12)
    plt.ylabel('', color=CINZA_CLARO)
    
    ax.tick_params(axis='y', colors=CINZA_CLARO, labelsize=12)
    ax.tick_params(axis='x', colors=CINZA_CLARO)
    ax.grid(axis='x', alpha=0.15, color=CINZA_CLARO)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)
    
    save_path_imp = os.path.join(plot_dir, 'feature_importance_final.png')
    plt.savefig(save_path_imp, dpi=150, facecolor=PRETO)
    
    csv_path_imp = os.path.join(plot_dir, 'feature_importance_dados.csv')
    df_imp.to_csv(csv_path_imp, index=False, sep=';', encoding='latin-1')
    plt.close()
    logger.info(f"✅ Feature Importance salvo em: {save_path_imp}")