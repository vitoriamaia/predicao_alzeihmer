# Predição de Conversão MCI para Demência

Projeto de Aprendizado de Máquina para predição de conversão de Comprometimento Cognitivo Leve (MCI) para Demência em 3 anos, utilizando dados do ADNI (Alzheimer's Disease Neuroimaging Initiative).

**Desenvolvido por:** Vitória

**Curso:** Aprendizado de Máquina e Mineração de Dados 2025.2

**Professor:** Leonardo Rocha

---

## Descrição do Problema

O Comprometimento Cognitivo Leve (MCI - Mild Cognitive Impairment) é uma condição intermediária entre o envelhecimento cognitivo normal e a demência. Identificar precocemente quais pacientes com MCI irão converter para demência é crucial para intervenção terapêutica antecipada.

Este projeto desenvolve modelos de aprendizado de máquina para predizer se um paciente com MCI irá converter para demência em um período de 3 anos, utilizando dados clínicos, cognitivos e de neuroimagem coletados na linha de base.

### Objetivo

Desenvolver um pipeline completo de Machine Learning incluindo:
- Análise exploratória e preparação de dados
- Engenharia de features e seleção de variáveis
- Treinamento e comparação de múltiplos modelos
- Otimização de hiperparâmetros
- Técnicas de ensemble
- Exportação de modelos em formato padrão (ONNX)
- API REST para servir os modelos em produção

### Dataset

**ADNI (Alzheimer's Disease Neuroimaging Initiative)**
- Dataset real de médio porte
- Pacientes diagnosticados com MCI (EMCI ou LMCI) na linha de base
- Acompanhamento longitudinal de 3 anos
- Features: dados demográficos, testes cognitivos, biomarcadores, volumes cerebrais
- Classes balanceadas: conversores vs não-conversores

---

## Estrutura do Repositório

```
predicao_alzeihmer/
│
├── data/                           # Dados do projeto
│   ├── ADNIMERGE_04Dec2025.csv    # Dataset original
│   ├── mci_pacientes_3anos_completo.csv
│   ├── mci_pacientes_baseline.csv
│   └── processed/                  # Dados processados para ML
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── feature_names.txt
│       ├── scaler.pkl
│       └── label_encoder.pkl
│
├── jupiter_notebooks/              # Notebooks de desenvolvimento
│   ├── data_preparation.ipynb     # 1. Preparação e limpeza dos dados
│   ├── data_exploration.ipynb     # 2. Análise exploratória
│   ├── 03_baseline_models.ipynb   # 3. Modelos baseline
│   ├── 04_advanced_models.ipynb   # 4. Modelos avançados
│   ├── 05_bayesian_optimization.ipynb  # 5. Otimização bayesiana
│   ├── 06_ensemble_weighted_rank.ipynb # 6. Ensemble methods
│   └── 07_export_models_onnx.ipynb     # 7. Exportação para ONNX
│
├── models/                         # Modelos treinados
│   ├── *.pkl                       # Modelos em formato pickle
│   └── onnx/                       # Modelos exportados em ONNX
│       ├── logistic_regression_optimized.onnx
│       ├── svm_rbf_optimized.onnx
│       ├── advanced_mlp_all.onnx
│       └── api_metadata.json
│
├── api/                            # API para model serving
│   ├── app.py                      # Aplicação FastAPI
│   └── README.md                   # Documentação da API
│
├── requirements.txt                # Dependências do projeto
└── README.md                       # Este arquivo
```

---

## Instalação e Configuração

### Pré-requisitos

- Python 3.8 ou superior
- pip ou conda para gerenciamento de pacotes

### Instalação das Dependências

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd predicao_alzeihmer

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Dependências Principais

- **Processamento de dados:** numpy, pandas, scipy
- **Visualização:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, xgboost, lightgbm
- **Otimização:** scikit-optimize
- **Exportação de modelos:** skl2onnx, onnx, onnxruntime
- **API:** fastapi, uvicorn, pydantic

---

## Reproduzindo os Experimentos

### 1. Preparação dos Dados

Execute o notebook de preparação de dados:

```bash
jupyter notebook jupiter_notebooks/data_preparation.ipynb
```

Este notebook realiza:
- Seleção de pacientes MCI com acompanhamento de 3 anos
- Limpeza e imputação de valores faltantes
- Codificação de variáveis categóricas
- Normalização de features
- Separação treino/teste estratificada
- Salvamento dos dados processados

### 2. Análise Exploratória

```bash
jupyter notebook jupiter_notebooks/data_exploration.ipynb
```

Análises incluídas:
- Distribuição das variáveis
- Análise temporal de conversores vs não-conversores
- Comparação de testes cognitivos (MMSE, ADAS)
- Análise de volumes cerebrais (hipocampo, ventrículos)
- Identificação de variáveis discriminativas

### 3. Treinamento de Modelos

Execute os notebooks de modelagem em sequência:

**a) Modelos Baseline:**
```bash
jupyter notebook jupiter_notebooks/03_baseline_models.ipynb
```
- Logistic Regression
- Decision Trees
- Random Forest
- Naive Bayes

**b) Modelos Avançados:**
```bash
jupyter notebook jupiter_notebooks/04_advanced_models.ipynb
```
- XGBoost
- LightGBM
- MLP (Multi-Layer Perceptron)
- SVM com diferentes kernels

**c) Otimização de Hiperparâmetros:**
```bash
jupyter notebook jupiter_notebooks/05_bayesian_optimization.ipynb
```
- RandomizedSearchCV com scipy distributions
- Feature selection com RFE
- Calibração de probabilidades
- Otimização de threshold

**d) Ensemble Methods:**
```bash
jupyter notebook jupiter_notebooks/06_ensemble_weighted_rank.ipynb
```
- Weighted Rank Averaging
- Combinação dos top modelos
- Validação cruzada

### 4. Exportação de Modelos para ONNX

```bash
jupyter notebook jupiter_notebooks/07_export_models_onnx.ipynb
```

Este notebook:
- Converte modelos sklearn para formato ONNX
- Valida as conversões
- Gera metadados para a API

---

## Executando a API

### Iniciar o Servidor

```bash
cd api
python app.py
```

Ou usando uvicorn diretamente:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### Acessar a Documentação Interativa

Após iniciar o servidor, acesse:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoints Disponíveis

**GET /**
- Informações básicas da API

**GET /health**
- Status de saúde e modelos carregados

**GET /models**
- Lista de modelos disponíveis

**GET /metadata**
- Metadados do problema e features

**POST /predict**
- Fazer predição para um único paciente

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, -0.3, ...],
    "model_name": "svm_rbf_optimized"
  }'
```

**POST /predict_batch**
- Fazer predições para múltiplos pacientes

### Exemplo de Uso em Python

```python
import requests

url = "http://localhost:8000/predict"

data = {
    "features": [0.5, 1.2, -0.3, 0.8, -1.1, 0.2, 0.9, -0.5, 
                 1.5, 0.1, -0.8, 0.6, -0.2, 1.0, 0.3, -1.2, 
                 0.7, -0.4, 0.9, 0.0],
    "model_name": "svm_rbf_optimized"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Predição: {result['class_label']}")
print(f"Probabilidade: {result['probability']:.2%}")
```

---

## Carregando Modelos Localmente

### Modelos ONNX

```python
import onnxruntime as rt
import numpy as np

# Carregar modelo
session = rt.InferenceSession("models/onnx/svm_rbf_optimized.onnx")

# Preparar dados
X_sample = np.array([[0.5, 1.2, -0.3, ...]], dtype=np.float32)

# Fazer predição
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
prediction = session.run([output_name], {input_name: X_sample})

print(prediction)
```

### Modelos Pickle (desenvolvimento)

```python
import pickle

# Carregar modelo
with open("models/svm_rbf_optimized.pkl", "rb") as f:
    model = pickle.load(f)

# Fazer predição
prediction = model.predict_proba(X_sample)[:, 1]
print(f"Probabilidade de conversão: {prediction[0]:.2%}")
```

### Carregar Pré-processadores

```python
import pickle

# Carregar scaler
with open("data/processed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Aplicar transformação
X_scaled = scaler.transform(X_raw)
```

---

## Resultados Principais

### Performance dos Modelos

| Modelo | CV ROC-AUC | Test ROC-AUC | F1-Score | Balanced Acc |
|--------|------------|--------------|----------|--------------|
| SVM RBF (otimizado) | 0.756 | 0.743 | 0.68 | 0.72 |
| MLP Advanced | 0.748 | 0.731 | 0.66 | 0.70 |
| Logistic Regression | 0.732 | 0.719 | 0.65 | 0.69 |
| Ensemble (Weighted Rank) | 0.761 | 0.749 | 0.70 | 0.73 |

### Variáveis Mais Importantes

1. MMSE (Mini-Mental State Examination)
2. ADAS-Cog13 (Alzheimer's Disease Assessment Scale)
3. Volume do Hipocampo
4. FAQ (Functional Activities Questionnaire)
5. CDR-SB (Clinical Dementia Rating Sum of Boxes)

### Métricas de Validação

- **Cross-Validation:** 10-fold Stratified, repeated 10 vezes
- **Métrica primária:** ROC-AUC
- **Métricas secundárias:** Accuracy, F1, Balanced Accuracy, Sensitivity, Specificity, PPV, NPV
- **Otimização de threshold:** Balanced Accuracy no conjunto de treino

---

## Pipeline Técnico Completo

### 1. Tratamento de Dados
- Imputação de valores faltantes (mediana para numéricos, moda para categóricos)
- Remoção de variáveis com >20% missing
- Encoding de variáveis categóricas (OneHotEncoder)
- Normalização (StandardScaler: média=0, desvio=1)

### 2. Engenharia de Features
- PCA para reduzir correlação entre variáveis ADAS
- Seleção de features:
  - Feature Set 1: Todas as features
  - Feature Set 2: Filtro estatístico (p<0.05)
  - Feature Set 5: RFE com Logistic Regression
  - Feature Set 7: RFE com Random Forest

### 3. Treinamento
- Modelos testados: 11 algoritmos diferentes
- Otimização: RandomizedSearchCV (40 iterações)
- Calibração de probabilidades (CalibratedClassifierCV)
- Ensemble: Weighted Rank Averaging dos top 3-5 modelos

### 4. Validação
- Repeated Stratified K-Fold (10 folds, 10 repetições)
- Hold-out test set estratificado (20%)
- Validação independente por site de coleta

---

## Tecnologias Utilizadas

- **Python 3.13**
- **scikit-learn 1.7.2** - Framework principal de ML
- **XGBoost 3.1.2** - Gradient boosting
- **LightGBM 4.6.0** - Gradient boosting otimizado
- **scikit-optimize 0.10.2** - Otimização bayesiana
- **ONNX 1.20.0** - Formato padrão de modelos
- **ONNXRuntime 1.23.2** - Inferência de modelos ONNX
- **FastAPI 0.124.4** - Framework web para API
- **Uvicorn 0.38.0** - Servidor ASGI
- **Pandas 2.3.3** - Manipulação de dados
- **NumPy 2.3.4** - Computação numérica
- **Matplotlib 3.10.7** - Visualização
- **Seaborn 0.13.2** - Visualização estatística

---

## Limitações e Trabalhos Futuros

### Limitações Identificadas

1. **Tamanho do dataset:** Conjunto relativamente pequeno pode limitar generalização
2. **Desbalanceamento:** Taxa de conversão de aproximadamente 30-35%
3. **Features temporais:** Não foram utilizadas informações longitudinais completas
4. **Modelos complexos:** XGBoost e LightGBM não foram convertidos para ONNX devido a incompatibilidades
5. **Validação externa:** Não foi testado em cohorts independentes

### Possíveis Melhorias

1. **Deep Learning:** Testar arquiteturas mais sofisticadas (LSTM, Transformers)
2. **Dados longitudinais:** Incorporar trajetória temporal das variáveis
3. **Neuroimagem:** Incluir features de ressonância magnética processadas
4. **Feature engineering avançada:** Interações entre variáveis, features polinomiais
5. **Ensemble mais sofisticado:** Stacking, Blending com meta-learners
6. **Calibração:** Métodos mais avançados (Isotonic Regression, Platt Scaling)
7. **Interpretabilidade:** SHAP values, LIME para explicação das predições
8. **Deploy:** Containerização com Docker, orquestração com Kubernetes

---

## Reprodutibilidade

Todos os experimentos são reproduzíveis através dos notebooks fornecidos. Os seeds aleatórios estão fixados em todos os pontos relevantes:

```python
random_state = 42
cv_random_seed = 1
```

Para reproduzir completamente:
1. Execute os notebooks na ordem numérica
2. Use o mesmo ambiente Python (requirements.txt)
3. Os dados processados já estão salvos em `data/processed/`

---

## Licença e Uso dos Dados

Este projeto utiliza dados do ADNI (Alzheimer's Disease Neuroimaging Initiative). O acesso aos dados ADNI requer aprovação e concordância com os termos de uso do consórcio ADNI.

**Importante:** Os dados ADNI não podem ser redistribuídos. Para reproduzir este projeto com os dados originais, é necessário:
1. Solicitar acesso ao ADNI: https://adni.loni.usc.edu/
2. Baixar o arquivo ADNIMERGE
3. Colocá-lo na pasta `data/`

---

## Contato e Contribuições

**Desenvolvedor:** Vitória

**Instituição:** Curso de Ciência da Computação

**Disciplina:** Aprendizado de Máquina e Mineração de Dados 2025.2

Para dúvidas ou sugestões sobre o projeto, entre em contato através do repositório.

---

## Referências

1. ADNI - Alzheimer's Disease Neuroimaging Initiative: https://adni.loni.usc.edu/
2. Petersen, R. C. (2004). Mild cognitive impairment as a diagnostic entity. Journal of Internal Medicine.
3. Scikit-learn documentation: https://scikit-learn.org/
4. ONNX documentation: https://onnx.ai/
5. FastAPI documentation: https://fastapi.tiangolo.com/

---

**Última atualização:** Dezembro 2025