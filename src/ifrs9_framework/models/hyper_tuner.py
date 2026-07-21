import optuna
import logging
import numpy as np
from sklearn.metrics import average_precision_score
from models.model_selector import ModelFactory

optuna.logging.set_verbosity(optuna.logging.ERROR) # Silenciar log original para UI limpa
logger = logging.getLogger("MLOps-Tuner")

class HyperparameterTuner:
    """
    Motor de otimização Bayesiana para encontrar os melhores hiperparâmetros
    focados em Precision-Recall AUC (ótimo para bases desbalanceadas de fraude).
    """

    def __init__(self, X_train, y_train, X_test, y_test, config, cat_features_idx=None, cat_cols=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.algo = self.config.get('algorithm', 'catboost').lower()
        self.cat_features_idx = cat_features_idx
        self.cat_cols = cat_cols
        
        # Como comunicamos o progresso para o HTML: injetando o tracker se ele existir
        self.tracker = None 

    def attach_tracker(self, tracker):
        """Associa o sistema de telemetria do Dataflow ao Optuna"""
        self.tracker = tracker

    def _objective(self, trial):
        search_space = self.config.get('hyperparameter_tuning', {}).get('search_space', {})
        
        # 1. Sugestões de parâmetros base
        lr_range = search_space.get('learning_rate', [0.01, 0.1])
        w_range = search_space.get('scale_pos_weight', [1.0, 15.0])
        
        params = {
            'learning_rate': trial.suggest_float('learning_rate', lr_range[0], lr_range[1]),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', w_range[0], w_range[1])
        }

        # 2. Sugestões específicas por algoritmo
        if self.algo == 'catboost':
            d_range = search_space.get('depth', [4, 8])
            params['depth'] = trial.suggest_int('depth', int(d_range[0]), int(d_range[1]))
            params['iterations'] = 50 # Menos iterações para o trial ser rápido
            
        elif self.algo == 'lightgbm':
            md_range = search_space.get('max_depth', [4, 8])
            params['max_depth'] = trial.suggest_int('max_depth', int(md_range[0]), int(md_range[1]))
            params['n_estimators'] = 50
            
        elif self.algo == 'ebm':
            mb_range = search_space.get('max_bins', [64, 512])
            inter_range = search_space.get('interactions', [0, 50])
            params['max_bins'] = trial.suggest_int('max_bins', int(mb_range[0]), int(mb_range[1]))
            params['interactions'] = trial.suggest_int('interactions', int(inter_range[0]), int(inter_range[1]))

        # 3. Instancia via Factory
        model = ModelFactory.get_model(self.algo, params, self.cat_features_idx)

        # 4. Tratamento Especial de Treino
        if self.algo == 'ebm':
            # EBM aplica pesos na amostra
            sample_w = np.where(self.y_train == 1, params.get('scale_pos_weight', 10.0), 1.0)
            model.fit(self.X_train, self.y_train, sample_weight=sample_w)
        else:
            model.fit(self.X_train, self.y_train)

        # 5. Avaliação (Otimizando para PR-AUC)
        preds = model.predict_proba(self.X_test)[:, 1]
        return average_precision_score(self.y_test, preds)

    def optimize(self, n_trials=5):
        logger.info(f"Iniciando Optuna para {self.algo.upper()} com {n_trials} trials.")
        study = optuna.create_study(direction='maximize')
        
        # O Logger visual do Dataflow
        def optuna_logger(study, trial):
            msg = f"[Optuna] Trial {trial.number + 1}/{n_trials} | PR-AUC = {trial.value:.4f} | LR = {trial.params.get('learning_rate', 0):.3f}"
            if self.tracker:
                # Usa 'running' para a caixa piscar em amarelo na tela HTML
                self.tracker.update_node("step_3", "running", msg)
            else:
                logger.info(msg)
                
        study.optimize(self._objective, n_trials=n_trials, callbacks=[optuna_logger])
        
        logger.info(f"Otimização concluída. Melhores parâmetros: {study.best_params}")
        return study.best_params