# API de Predição - Conversão MCI para Demência

API REST desenvolvida com FastAPI para servir modelos de predição de conversão de MCI (Mild Cognitive Impairment) para Demência em 3 anos.

## Instalação

```bash
pip install fastapi uvicorn onnxruntime scikit-learn numpy pydantic
```

## Executar a API

```bash
cd api
python app.py
```

Ou usando uvicorn diretamente:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em: `http://localhost:8000`

## Documentação Interativa

Após iniciar a API, acesse:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### GET /
Informações básicas da API

### GET /health
Status de saúde da API e modelos carregados

### GET /models
Lista de modelos disponíveis

### GET /metadata
Metadados do problema e dos modelos

### POST /predict
Fazer predição para um único paciente

**Request Body:**
```json
{
  "features": [valor1, valor2, ..., valorN],
  "model_name": "nome_do_modelo"  // opcional
}
```

**Response:**
```json
{
  "model_used": "nome_do_modelo",
  "prediction": 0,
  "probability": 0.23,
  "class_label": "Nao-Conversor"
}
```

### POST /predict_batch
Fazer predições para múltiplos pacientes

## Exemplo de Uso

### Python
```python
import requests

url = "http://localhost:8000/predict"

data = {
    "features": [0.5, 1.2, -0.3, ...],  # 20+ features
    "model_name": "elastic_net"
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, -0.3, ...]}'
```

## Estrutura de Features

As features devem estar na mesma ordem e escala dos dados de treinamento. 
Consulte `feature_names.txt` para a lista completa.

## Modelos Disponíveis

Os modelos são carregados automaticamente do diretório `../models/onnx/`.
Use o endpoint `/models` para ver quais estão disponíveis.
