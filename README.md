
# Integração da Interface MLOps na Arquitetura

O arquivo que geramos na nossa última iteração chama-se  **`config_ui.html`** . Ele é a materialização da sua ideia de um "Painel Self-Service" para o Cientista de Dados.

Aqui estão as respostas detalhadas para as suas dúvidas sobre como vamos integrar isso na nossa esteira.

### 1. Onde colocar o arquivo na nossa Arquitetura?

Para mantermos as diretrizes de código limpo, não vamos misturar arquivos HTML com scripts de Machine Learning. Vamos criar uma nova pasta na raiz do projeto chamada `ui/` ou `frontend/`. A nossa árvore ficará assim:

```
ifrs9-financial-computer-ai/
│
├── src/                      # Scripts Python (features, models, orchestrator)
├── config/                   # Onde mora o .env e o config.yaml
│
├── ui/                       # A NOVA PASTA PARA A INTERFACE
│   ├── config_ui.html        # (O painel de configuração que criamos)
│   ├── vertex_pipeline.html  # (O visualizador do Dataflow que criamos antes)
│   └── dashboard_results.html# (O futuro painel com gráficos SHAP, ROC, etc.)
│
├── docker-compose.yml
└── Dockerfile
```

### 2. Pode ser tudo num único arquivo HTML?

**Sim, absolutamente.** A abordagem que usamos (HTML puro + Tailwind via CDN + JavaScript embutido) chama-se  *Single Page Application (SPA) Monolítica* .

Para um painel interno de um banco focado em MLOps, essa é a melhor abordagem. Você não precisa de configurações complexas de Node.js, React ou Angular. O arquivo é leve, rápido e qualquer membro da equipe pode abri-lo diretamente no navegador para testar o layout.

### 3. Como vai funcionar a interação? Ele lê/escreve o `.yml` sozinho?

Aqui está o grande "pulo do gato" da engenharia de software. **O HTML rodando no seu navegador (Frontend) não tem permissão de segurança para salvar ou ler arquivos diretamente no disco do seu computador.** Portanto, o `config_ui.html` sozinho não consegue alterar o `config.yaml`. Para fazer essa mágica acontecer, nós não precisamos de um Banco de Dados novo; nós precisamos de uma  **API "Ponte"** .

**O Fluxo Profissional (O que vamos construir):**

1. **O Cientista usa a Tela:** Você entra no `config_ui.html`, seleciona as colunas, marca o Optuna e clica em "Salvar Configuração".
2. **O HTML faz um POST (Envio):** O JavaScript do nosso HTML pega essas opções e envia via rede (uma requisição HTTP POST).
3. **A Ponte em Python (FastAPI):** Vamos adicionar um micro-servidor muito leve em Python (usando a biblioteca `FastAPI` ou `Flask` dentro do nosso Docker). Esse servidor fica "escutando".
4. **Python escreve o YAML:** O `FastAPI` recebe os dados do HTML e **ele sim** (pois é um script rodando no servidor) sobrescreve o arquivo `config.yaml` real no disco.
5. **Gatilho de Execução:** Logo após salvar o YAML, o próprio `FastAPI` pode rodar um comando chamando o nosso `pipeline_orchestrator.py` para iniciar o treinamento.

### Precisamos adicionar um Banco de Dados (BD) por trás?

Para guardar as configurações:  **NÃO** .
O próprio arquivo `config.yaml` é a nossa "fonte da verdade". Usar arquivos YAML como banco de dados de configuração é o que chamamos de *Infrastructure as Code (IaC)* ou, no nosso caso,  *ML as Code* .

Para guardar o histórico de treinamentos e os logs:  **NÃO PRECISAMOS DE UM NOVO** , pois nós já temos o **MLflow** (`mlflow.db`) rodando no seu Docker! O MLflow é o nosso banco de dados oficial de MLOps.

### Resumo do Próximo Passo de Engenharia

Nossa interface visual (`config_ui.html`) está pronta esteticamente. O nosso motor (`pipeline_orchestrator.py`) está pronto matematicamente.

O que falta para conectarmos os dois mundos é criarmos um arquivo chamado `api.py` (usando FastAPI). Ele será o carteiro que pega as opções que o Cientista clicou no HTML e escreve no `config.yaml` para o Orquestrador rodar.
