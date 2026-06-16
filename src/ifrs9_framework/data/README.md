# Gerador Sintetico Arificial de Base Risco de Crédito (simulando dados reais com base em matematica e fisica)

1. Assimetria e Caudas Pesadas (Lei de Potência)
A distribuição de riqueza no Brasil (e no mundo) não é um sino perfeito. Ela é brutalmente desigual.

O Código:
```python
## Python
renda_pareto = (np.random.pareto(a=1.16, size=self.n_clientes) + 1) * 1320
```

#### **A Matemática:**

Aqui usamos a Distribuição de Pareto. Na física e na economia, ela modela eventos onde uma pequena quantidade de instâncias concentra a maior parte do volume (o famoso princípio 80/20). A Função Densidade de Probabilidade (PDF) é:

![Densidade de Probabilidade](/src/ifrs9_framework/data/img/pdf.png)

* **Onde** $\alpha = 1.16$ é o índice de decaimento (simulando um índice de Gini alto) e $x_m$ é a nossa base (1320, o salário mínimo).

**Por que usar?**
Se usássemos uma distribuição Normal para a renda, teríamos aposentados ganhando valores negativos ou uma concentração irreal em torno de R$ 5.000. O Pareto cria a "Cauda Pesada" (Fat Tail): 90% da base se espreme perto do salário mínimo, e uma microfração ganha R$ 150.000. É isso que cria a assimetria que "quebra" os modelos lineares tradicionais.

2. Ruído Branco Multivariado (Adicionando Entropia)
O universo não é determinístico. As variáveis não se movem em linhas retas isoladas.

O Código:
```python
## Python
base_limite = df_contratos['renda'] * np.random.lognormal(mean=0.2, sigma=0.6)
ruido_limite = np.random.normal(0, 1000)
limite_sujo = np.maximum(0, base_limite + ruido_limite)
```


#### **A Matemática:**

Aqui aplicamos Ruído Estocástico $\epsilon$.
* Primeiro, não multiplicamos a renda por um número fixo. Multiplicamos por uma variável extraída de uma distribuição Lognormal. O Lognormal garante que o multiplicador nunca será negativo (não existe limite de crédito negativo), mas permite picos abruptos para a direita.
* Depois, somamos um Erro Gaussiano (Normal): $\epsilon \sim \mathcal{N}(0, 1000^2)$. Isso significa que adicionamos ou subtraímos aleatoriamente valores em torno de zero, com um desvio padrão de R$ 1.000.
* **Por que usar?** É esse $\epsilon$ que transforma "raios laser" em "nuvens" no nosso gráfico PCA. Ele simula o componente irracional humano e as exceções da política de crédito do banco.

## 
3. As Famílias Gama ($\Gamma$) e Beta ($\beta$)

* Em finanças, os limites importam. Valores de empréstimo não podem ser negativos, e taxas de utilização de cartão não podem passar de 100% (idealmente).

O Código:
```python
## Python
# Empréstimos (Sempre positivos, assimétricos)
valor_financiado_gama = np.random.gamma(shape=2.5, scale=scale_gama)

# Uso de Cartão de Crédito (Limitado entre 0 e 1, ou 0% a 100%)
uso_cartao = np.random.beta(a=1.5, b=5.0)
```

#### **A Matemática:**

* Distribuição Gama ($\Gamma$): Modelada pelos parâmetros de forma ($k$ ou shape) e escala ($\theta$ ou scale). Ela só existe para valores maiores que zero. Perfeita para modelar o valor exato de um contrato de empréstimo pessoal.
* Distribuição Beta ($\beta$): É a rainha das proporções. Ela é contida no intervalo estrito $[0, 1]$. Ao definirmos a=1.5 e b=5.0 (onde $b > a$), forçamos a curva a se concentrar perto de $0.2$. Ou seja, simulamos que a maioria dos clientes usa cerca de 20% do limite do cartão, mas alguns poucos chegam a 100%.

##
4. A Física do Risco (Regressão Logística e Função Sigmoide)
* Como transformar dezenas de colunas em uma probabilidade de calote contida entre 0% e 100%?

O Código:
```python
## Python
z = -4.8 - 0.001 * df['bureau_score'] + ...
prob_default = 1 / (1 + np.exp(-z))
```


#### **A Matemática:**

* Essa é a base do Modelo Estrutural de Risco. Primeiro, calculamos $z$, que é uma combinação linear (uma soma de forças vetoriais) que empurra o cliente para longe ou para perto do abismo financeiro. O intercepto ($-4.8$) é a "gravidade" da carteira de consignado que puxa todo mundo para a adimplência.
* Para transformar $z$ (que pode ir de $-\infty$ a $+\infty$) em uma probabilidade real $P \in [0, 1]$, passamos $z$ pela Função Sigmoide:

![Função Sigmoide](/src/ifrs9_framework/data/img/sigmoid.png)


#### **Documentação Aprofundada:** [Estrutura Financeira](/src/ifrs9_framework/data/docs/Estrutura-Financeira.docx)
