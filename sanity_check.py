import torch
import mlflow
from catboost import CatBoostClassifier
import numpy as np

def run_sanity_check():
    print("Iniciando Verificação de Sanidade da Infraestrutura de IA...\n")

    # 1. Teste do PyTorch e CUDA (GPU)
    print("--- [1/3] Verificando PyTorch e CUDA ---")
    print(f"Versão do PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Disponível? {'SIM' if cuda_available else 'NÃO'}")
    
    if cuda_available:
        print(f"Nome da GPU Detectada: {torch.cuda.get_device_name(0)}")
        print(f"Total de Memória da GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("ALERTA: PyTorch não detectou a GPU. O treinamento será feito na CPU.\n")

    # 2. Teste do CatBoost com aceleração de GPU
    print("--- [2/3] Verificando CatBoost com task_type='GPU' ---")
    try:
        X_dummy = np.random.rand(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        
        task_type = 'GPU' if cuda_available else 'CPU'
        
        # O pulo do gato para o WSL2: limitamos o uso da VRAM a 50% 
        # para evitar concorrência com o Windows Host
        model = CatBoostClassifier(
            iterations=10, 
            task_type=task_type,
            devices='0',       # Força o uso da GPU 0
            gpu_ram_part=0.25,  # Usa apenas 50% da memória de vídeo livre
            verbose=0
        )
        model.fit(X_dummy, y_dummy)
        print(f"CatBoost treinou com sucesso usando: {task_type}\n")
    except Exception as e:
        print(f"Erro ao testar CatBoost: {e}\n")

    # 3. Teste do servidor MLflow
    print("--- [3/3] Verificando Comunicação com o MLflow ---")
    try:
        # Aponta para o servidor que subimos via Docker Compose
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Setup_Sanity_Check")
        
        with mlflow.start_run():
            mlflow.log_param("test_param", "OK")
            mlflow.log_metric("test_metric", 0.99)
            
        print("MLflow: Experimento registrado com sucesso! Acesse http://localhost:5000 no seu navegador.\n")
    except Exception as e:
        print(f"Erro ao conectar com MLflow: {e}\n")

    print("Verificação Finalizada.")

if __name__ == "__main__":
    run_sanity_check()