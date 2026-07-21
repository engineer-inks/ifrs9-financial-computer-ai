import os
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from core_utils import DOMAIN_MAPPINGS

logger = logging.getLogger("IFRS9_Engine.Features")

def get_focal_loss_obj(alpha_val, gamma_val):
    class FocalLossObjective(object):
        def calc_ders_range(self, approxes, targets, weights):
            probs = 1.0 / (1.0 + np.exp(-np.clip(approxes, -15, 15)))
            p_t = np.where(targets == 1, probs, 1 - probs)
            alpha_t = np.where(targets == 1, alpha_val, 1 - alpha_val)
            grad = alpha_t * (targets - probs) * (1 - p_t)**gamma_val
            hess = alpha_t * (1 - p_t)**gamma_val * probs * (1 - probs)
            hess = np.maximum(hess, 1e-4)
            return [(-grad[i], hess[i]) for i in range(len(targets))]
    return FocalLossObjective()

def converter_para_string(x):
    return x.astype(str)

def safe_col(df, col_name, default_val=0.0):
    if col_name in df.columns:
        return df[col_name]
    return pd.Series(default_val, index=df.index)

def load_and_filter_data(config, logger):
    data_paths = config.get('data_paths', {})
    raw_path = data_paths.get('raw_data') if isinstance(data_paths, dict) else None
    if not raw_path:
        raw_path = config.get('data_path', "../src/ifrs9_framework/data/raw/synthetic_credit_data.parquet")
    
    if raw_path and raw_path.startswith("../"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        raw_path = os.path.normpath(os.path.join(base_dir, raw_path.replace("../src/ifrs9_framework/", "")))
        
    if not raw_path or not os.path.exists(raw_path):
        raw_path = "../src/ifrs9_framework/data/raw/synthetic_credit_data.parquet"
        if not os.path.exists(raw_path):
            raw_path = "src/ifrs9_framework/data/raw/synthetic_credit_data.parquet"
        
    df = pd.read_parquet(raw_path)
    
    # GARANTE QUE TODAS AS COLUNAS SÃO STRINGS EXPLICITAMENTE
    df.columns = [str(c) for c in df.columns]
    
    product_codes = config.get('product_codes')
    if product_codes and 'IDT_PDT' in df.columns:
        df = df[df['IDT_PDT'].isin(product_codes)]
    logger.info(f"Dados carregados com segurança: {df.shape[0]} linhas.")
    return df

def features_engineer(df, config, logger):
    logger.info("Criando features comportamentais e financeiras...")
    df_eng = df.copy()
    
    # CONVERSÃO ABSOLUTA DE TODAS AS COLUNAS PARA STRING ANTES DE QUALQUER REPLACE
    cleaned_columns = []
    for c in df_eng.columns:
        col_str = str(c).replace("'", "").strip()
        cleaned_columns.append(col_str)
    df_eng.columns = cleaned_columns
    
    date_cols = ['DTA_INI_OPR', 'DTA_RFC', 'DTA_NAS']
    for col in date_cols:
        if col in df_eng.columns:
            df_eng[col] = pd.to_datetime(df_eng[col], errors='coerce')
            
    if 'IDT_EPC_BNF' in df_eng.columns:
        mapping_especie = config.get('mapping_especie', DOMAIN_MAPPINGS.get('mapping_especie', {}))
        df_eng['NEW_GRP_EPC_BNF'] = df_eng['IDT_EPC_BNF'].map(mapping_especie).fillna('OUTROS')
        
    if 'QTD_DIA_VCD' in df_eng.columns: df_eng['QTD_DIA_VCD'] = safe_col(df_eng, 'QTD_DIA_VCD', 0).clip(lower=0)
    if 'PZO_RMN' in df_eng.columns: df_eng['PZO_RMN'] = safe_col(df_eng, 'PZO_RMN', -1).clip(lower=-1)
    if 'TMP_RLN' in df_eng.columns: df_eng['TMP_RLN'] = safe_col(df_eng, 'TMP_RLN', 0).clip(lower=0)
    
    if 'TMP_CMO_BNF_DIA' in df_eng.columns:
        tmp_val = safe_col(df_eng, 'TMP_CMO_BNF_DIA', 0)
        df_eng['FLAG_TMP_CMO'] = np.where((tmp_val < 0) | (tmp_val > 20000), 1, 0)
        
    sld_ctb = safe_col(df_eng, 'SLD_CTB_LQD_PVSCLI', 0.0)
    vlr_ren = safe_col(df_eng, 'VLR_REN', 0.0)
    gra_vul = safe_col(df_eng, 'GRA_VUL', 0.0)
    idd_opr = safe_col(df_eng, 'IDD_DTA_OPR', 0.0)
    qtd_vcd = safe_col(df_eng, 'QTD_DIA_VCD', 0.0)
    qtd_pcl_vcd = safe_col(df_eng, 'QTD_PCL_VCD', 0.0)
    qtd_pcl_tot = safe_col(df_eng, 'QTD_PCL_TOT', 1.0)
    qtd_pcl_pag = safe_col(df_eng, 'QTD_PCL_PAG', 0.0)
    qtd_opr_vcd = safe_col(df_eng, 'QTD_OPR_VCD', 0.0)
    qtd_opr_tot = safe_col(df_eng, 'QTD_OPR_TOT', 1.0)
    tax_eft = safe_col(df_eng, 'TAX_EFT_ANO', 0.0)
    tmp_rln = safe_col(df_eng, 'TMP_RLN', 0.0)
    pzo_rmn = safe_col(df_eng, 'PZO_RMN', 1.0)

    df_eng['EXPOSICAO_SOBRE_RENDA'] = sld_ctb / (vlr_ren + 0.1)
    df_eng['RESILIENCIA_FINANCEIRA'] = gra_vul / (vlr_ren + 0.1)
    df_eng['MATURIDADE_FINANCEIRA'] = vlr_ren / (idd_opr.replace(0, 1) + 0.1)
    
    df_eng['DIAS_VENCIDOS_PARCELA'] = qtd_vcd / (qtd_pcl_vcd + 0.1)
    df_eng['PROP_PCL_VCD'] = qtd_pcl_vcd / qtd_pcl_tot.replace(0, 1)
    df_eng['PROP_PCL_PAG'] = qtd_pcl_pag / qtd_pcl_tot.replace(0, 1)
    df_eng['PROP_OPER_VCD'] = qtd_opr_vcd / qtd_opr_tot.replace(0, 1)
    
    df_eng['GRA_VUL_IDADE'] = gra_vul / (idd_opr + 0.1)
    df_eng['IDC_SUSTENTABILIDADE_ENCARGOS'] = vlr_ren / (tax_eft + 0.1)
    df_eng['CONFIABILIDADE'] = tmp_rln / (tax_eft + 0.1)
    df_eng['SALDO_POR_VUL'] = sld_ctb / (gra_vul + 0.1)
    df_eng['RENDA_SOBRE_PRAZO'] = vlr_ren / (pzo_rmn + 0.1)
    df_eng['RENDA_SOBRE_OPR_TOT'] = vlr_ren / (qtd_opr_tot + 0.1)
    
    idt_stu = safe_col(df_eng, 'IDT_STU_BNF', 0)
    df_eng['BENEFICIO_ATIVO'] = np.where(idt_stu == 3, 1, 0)
    df_eng['IDADA_QUAD'] = idd_opr ** 2
    df_eng['FLAG_SEM_ATRASO'] = np.where(qtd_vcd == 0, 1, 0)
    
    if 'TOTAL_DIAS_ATRASO_ACUMULADO' in df_eng.columns:
        tot_atraso = safe_col(df_eng, 'TOTAL_DIAS_ATRASO_ACUMULADO', 0.0)
        df_eng['MEDIA_DIAS_EM_ATRASO'] = tot_atraso / (qtd_pcl_pag + 0.1)
        
    if 'DTA_RFC' in df_eng.columns and 'DTA_NAS' in df_eng.columns:
        df_eng['FLAG_ANIVERSARIO'] = np.where(df_eng['DTA_RFC'].dt.month == df_eng['DTA_NAS'].dt.month, 1, 0)
        
    df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
    logger.info("Engenharia de features concluída com sucesso.")
    return df_eng

def build_preprocessor(df, config, logger):
    base_binary = config.get('binary_features', [])
    base_categorical = config.get('categorical_features', [])
    target = config.get('target_column', 'default_flag')
    
    binary_features = [f for f in base_binary if f in df.columns]
    categorical_features = [f for f in base_categorical if f in df.columns]
    
    num_cols = df.select_dtypes(include=np.number).columns
    numeric_features = [f for f in num_cols if f not in binary_features + categorical_features + [target]]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NAO_INFORMADO')),
        ('to_string', FunctionTransformer(converter_para_string, validate=False, feature_names_out='one-to-one'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', binary_transformer, binary_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')
    
    return preprocessor