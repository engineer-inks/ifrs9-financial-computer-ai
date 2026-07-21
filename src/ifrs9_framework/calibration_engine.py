import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger("IFRS9_Engine.Calibration")

# ==========================================
# 1. SHIFT MATEMÁTICO (REVERSÃO DE VIÉS)
# ==========================================
def recalibrar_probabilidade_shift(prob_modelo, peso_usado):
    """
    Reverte o viés introduzido pelo scale_pos_weight (ou class_weight).
    Fórmula de ajuste de odds baseada na proporção de amostragem.
    Garante que a média da PD seja idêntica à média da base real.
    """
    # Adicionando um pequeno epsilon para evitar erros de divisão por zero (Inf/NaN)
    epsilon = 1e-15
    prob_modelo = np.clip(prob_modelo, epsilon, 1 - epsilon)
    
    return prob_modelo / (prob_modelo + (1 - prob_modelo) * peso_usado)

# ==========================================
# 2. GRÁFICOS EXECUTIVOS (RELIABILITY)
# ==========================================
def gerar_analise_calibracao(y_true, y_prob_final, output_dir, label_plot, colors=None):
    """
    Gera o Reliability Diagram (Gráfico de Calibração) no padrão visual do Banco.
    Ajuda o comité a entender se o modelo está sub/super-estimando o risco.
    """
    logger.info(f"Gerando gráfico de calibração para o plot: {label_plot}...")
    
    # Cores padrão caso o core_utils falhe
    c_bg = colors.get('bg', '#000000') if colors else '#000000'
    c_primary = colors.get('primary', '#1526FF') if colors else '#1526FF'
    c_target = colors.get('target', '#FF007F') if colors else '#FF007F'
    c_grid = colors.get('grid', '#CCCCCC') if colors else '#CCCCCC'
    
    # Cálculo das médias por faixa (Decis)
    prob_true, prob_pred = calibration_curve(y_true, y_prob_final, n_bins=10, strategy='quantile')
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6), facecolor=c_bg)
    ax = plt.gca()
    ax.set_facecolor(c_bg)
    
    # Linha principal do Modelo
    plt.plot(prob_pred, prob_true, marker='o', linewidth=3, color=c_primary, label=f'Modelo: {label_plot}')
    
    # Referência (Calibração Perfeita / Diagonal 45º)
    plt.plot([0, 1], [0, 1], linestyle='--', color=c_target, alpha=0.7, label='Calibração Perfeita (1:1)')
    
    # Estilização Padrão Executivo
    plt.xlabel('PD Predita Final (Média da Faixa)', color=c_grid, fontsize=11)
    plt.ylabel('Taxa Real de Default (Observada)', color=c_grid, fontsize=11)
    plt.title(f'Reliability Diagram: {label_plot}', color='white', fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(facecolor=c_bg, edgecolor=c_grid, fontsize=10)
    plt.grid(True, alpha=0.2, color=c_grid)
    ax.tick_params(colors=c_grid, labelsize=10)
    
    # Remoção das bordas superior e direita para visual mais limpo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(c_grid)
    ax.spines['left'].set_color(c_grid)
    
    # Salvamento
    safe_label = label_plot.lower().replace(" ", "_").replace("/", "_")
    save_path = os.path.join(output_dir, f'calibracao_diagrama_{safe_label}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=c_bg, dpi=130)
    plt.close()
    
    logger.debug(f"Gráfico de calibração guardado com sucesso: {save_path}")