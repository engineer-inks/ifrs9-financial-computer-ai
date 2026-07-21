import logging
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

logger = logging.getLogger("MLOps-ModelSelector")

class ModelFactory:
    """
    Fábrica responsável por instanciar os modelos corretos com base na string solicitada
    e realizar adaptações específicas (ex: suporte nativo a categóricas no LightGBM vs CatBoost).
    """

    @staticmethod
    def get_model(algo_name: str, params: dict, cat_features_idx: list = None):
        """
        Retorna a instância do modelo configurada.
        :param algo_name: 'catboost', 'lightgbm' ou 'ebm'
        :param params: Hiperparâmetros
        :param cat_features_idx: Lista com os índices das colunas categóricas (usado no CatBoost)
        """
        algo = algo_name.lower()
        logger.info(f"Fábrica instanciando modelo: {algo.upper()}")

        if algo == 'catboost':
            # Removemos keys que não pertencem ao CatBoost
            params.pop('max_depth', None)
            params.pop('num_leaves', None)
            params.pop('max_bins', None)
            params.pop('interactions', None)

            return CatBoostClassifier(
                **params,
                cat_features=cat_features_idx if cat_features_idx else [],
                verbose=0,
                random_seed=42
            )

        elif algo == 'lightgbm':
            # Removemos keys exclusivas de outros algoritmos
            params.pop('depth', None)
            params.pop('l2_leaf_reg', None)
            params.pop('max_bins', None)
            params.pop('interactions', None)

            return LGBMClassifier(
                **params,
                random_state=42,
                verbose=-1
            )

        elif algo == 'ebm':
            # EBM tem um set de parâmetros muito específico
            params.pop('depth', None)
            params.pop('l2_leaf_reg', None)
            params.pop('iterations', None)
            params.pop('max_depth', None)
            params.pop('num_leaves', None)
            
            # Removemos o scale_pos_weight porque o EBM usa sample_weight no .fit()
            params.pop('scale_pos_weight', None) 

            return ExplainableBoostingClassifier(
                **params,
                random_state=42
            )

        else:
            raise ValueError(f"Algoritmo não suportado: {algo_name}. Escolha: catboost, lightgbm ou ebm.")