import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# --- BIBLIOTECAS DE ML ---
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, brier_score_loss, accuracy_score, confusion_matrix, matthews_corrcoef
from scipy.stats import ks_2samp
import catboost as cb
import joblib

try:
    from betacal import BetaCalibration
except ImportError:
    BetaCalibration = None

# ==========================================
# 1. MOTOR DE LOGGING ROBUSTO
# ==========================================
def configurar_logging(output_dir="logs"):
    """Inicializa o logger padrão capturando eventos tanto no terminal quanto em arquivo."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "execucao_modelo.log")
    
    logger = logging.getLogger("IFRS9_Engine")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

# ==========================================
# 2. IDENTIDADE VISUAL E GRÁFICOS (Padrão Banco)
# ==========================================
AZUL_PRIMARIO = "#1526FF"
AZUL_SECUNDARIO = "#0066FC"
PRETO = "#000000"
CINZA_ESCURO = "#333333" 
CINZA_CLARO = "#CCCCCC"
AZUL_CLARO = "#00D6FF"
AMARELO = "#FFF028"
ROSA = "#FF007F"

BANK_COLORS = [AZUL_PRIMARIO, AZUL_SECUNDARIO, PRETO, CINZA_ESCURO, CINZA_CLARO, AZUL_CLARO, AMARELO, ROSA]

def apply_custom_plot_style():
    """Aplica o padrão visual corporativo aos gráficos matplotlib/seaborn."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=BANK_COLORS),
        'axes.edgecolor': CINZA_ESCURO,
        'grid.color': CINZA_CLARO,
        'figure.facecolor': PRETO
    })

# ==========================================
# 3. DICIONÁRIOS DE DOMÍNIO (Regras de Negócio Estáticas)
# ==========================================
# Nota: product_codes, target_column e as features vêm do config gerado pelo UI.
# Mantemos aqui apenas os depara de domínio que não variam.
DOMAIN_MAPPINGS = {
    'mapping_especie': {
        10: 'APOSENTADORIA_POR_IDADE', 11: 'APOSENTADORIA_POR_IDADE', 39: 'APOSENTADORIA_POR_IDADE',
        7: 'APOSENTADORIA_POR_INVALIDEZ', 9: 'APOSENTADORIA_POR_INVALIDEZ', 31: 'APOSENTADORIA_POR_INVALIDEZ',
        40: 'APOSENTADORIA_TEMPO_CONTRIBUICAO', 43: 'APOSENTADORIA_TEMPO_CONTRIBUICAO', 44: 'APOSENTADORIA_TEMPO_CONTRIBUICAO', 55: 'APOSENTADORIA_TEMPO_CONTRIBUICAO',
        5: 'BENEFICIOS_ACIDENTARIOS', 8: 'BENEFICIOS_ACIDENTARIOS', 77: 'BENEFICIOS_ACIDENTARIOS', 90: 'BENEFICIOS_ACIDENTARIOS', 91: 'BENEFICIOS_ACIDENTARIOS',
        20: 'ENCARGOS_PREVIDENCIARIOS_UNIAO', 24: 'ENCARGOS_PREVIDENCIARIOS_UNIAO', 57: 'ENCARGOS_PREVIDENCIARIOS_UNIAO',
        1: 'PENSAO_POR_MORTE', 3: 'PENSAO_POR_MORTE', 21: 'PENSAO_POR_MORTE', 23: 'PENSAO_POR_MORTE', 27: 'PENSAO_POR_MORTE', 28: 'PENSAO_POR_MORTE', 55: 'PENSAO_POR_MORTE', 84: 'PENSAO_POR_MORTE'
    }
}