import matplotlib.pyplot as plt
import seaborn as plt_sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """
    Módulo Self-Service de visualização de métricas de modelo de risco.
    Gera o pacote completo de auditoria para o Banco Central/Comitê.
    """
    def __init__(self, y_true, y_pred_proba, y_pred_class, style='dark_background'):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred_class = y_pred_class
        plt.style.use(style)

    def plot_roc_and_ks(self):
        """Gera a Curva ROC e calcula o Teste de Kolmogorov-Smirnov (KS)"""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Cálculo do KS
        ks_stat = np.max(tpr - fpr)
        ks_idx = np.argmax(tpr - fpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#00ffcc', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        # Desenhando a linha do KS
        ax.plot([fpr[ks_idx], fpr[ks_idx]], [fpr[ks_idx], tpr[ks_idx]], 
                color='#ff3366', linestyle=':', lw=2, label=f'KS = {ks_stat*100:.1f}%')
        
        ax.set_title('Poder de Discriminação: Curva ROC & Estatística KS')
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, nota_corte):
        """Gera a Matriz de Confusão para análise de lucratividade"""
        cm = confusion_matrix(self.y_true, self.y_pred_class)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt_sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Aprovado (0)', 'Negado (1)'],
                        yticklabels=['Bom Pagador (0)', 'Inadimplente (1)'])
        ax.set_title(f'Matriz de Confusão e Impacto Comercial (Corte: {nota_corte*100:.1f}%)')
        ax.set_ylabel('Realidade (Mundo Real)')
        ax.set_xlabel('Decisão do Modelo')
        plt.tight_layout()
        return fig

    def plot_hosmer_lemeshow(self, bins=10):
        """Gera a curva de calibração para garantir o provisionamento correto"""
        prob_real, prob_prevista = calibration_curve(self.y_true, self.y_pred_proba, n_bins=bins, strategy='quantile')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_prevista, prob_real, marker='o', color='#ff3366', lw=2, label='Calibração do Modelo')
        ax.plot([0, prob_real.max()], [0, prob_real.max()], linestyle='--', color='gray', label='Calibração Perfeita')
        
        ax.set_title('Gráfico de Calibração (Hosmer-Lemeshow)')
        ax.set_xlabel('Probabilidade Média Prevista')
        ax.set_ylabel('Taxa Real de Inadimplência Observada')
        ax.legend()
        plt.tight_layout()
        return fig

    def generate_full_audit_panel(self, nota_corte=0.04):
        """Orquestra a geração de todos os gráficos e relatórios."""
        logger.info("Gerando Painel de Auditoria Self-Service...")
        
        # Mostra o relatório de classificação em texto
        print("\n--- RELATÓRIO DE CLASSIFICAÇÃO COMERCIAL ---")
        print(classification_report(self.y_true, self.y_pred_class, target_names=['Bons Pagadores', 'Inadimplentes']))
        
        # Plota os 3 gráficos principais
        self.plot_roc_and_ks()
        self.plot_confusion_matrix(nota_corte)
        self.plot_hosmer_lemeshow()
        plt.show()