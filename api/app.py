from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import onnxruntime as rt
import pickle
import json
from pathlib import Path

app = FastAPI(
    title="API Predicao Alzheimer MCI",
    description="API para predicao de conversao de MCI para Demencia usando modelos treinados",
    version="1.0.0"
)

ONNX_DIR = Path("../models/onnx/")
DATA_DIR = Path("../data/processed/")

with open(ONNX_DIR / "api_metadata.json", "r") as f:
    metadata = json.load(f)

with open(DATA_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

sessions = {}
for model_name in metadata['models_available']:
    model_path = ONNX_DIR / f"{model_name}.onnx"
    if model_path.exists():
        try:
            sessions[model_name] = rt.InferenceSession(str(model_path))
            print(f"Modelo {model_name} carregado com sucesso")
        except Exception as e:
            print(f"Erro ao carregar {model_name}: {str(e)[:100]}")
            continue

if len(sessions) == 0:
    raise RuntimeError("Nenhum modelo foi carregado com sucesso")

print(f"\nTotal de modelos carregados: {len(sessions)}")
print(f"Modelos disponiveis: {list(sessions.keys())}")

class PredictionInput(BaseModel):
    features: List[float] = Field(..., description="Lista de features do paciente")
    model_name: Optional[str] = Field(None, description="Nome do modelo a usar")

class PredictionOutput(BaseModel):
    model_used: str
    prediction: int
    probability: float
    class_label: str

@app.get("/")
def root():
    return {
        "message": "API de Predicao de Conversao MCI para Demencia",
        "status": "online",
        "models_available": len(sessions),
        "endpoints": ["/predict", "/models", "/health", "/metadata"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(sessions),
        "models": list(sessions.keys())
    }

@app.get("/models")
def list_models():
    return {
        "available_models": list(sessions.keys()),
        "default_model": metadata['best_model'],
        "total": len(sessions)
    }

@app.get("/metadata")
def get_metadata():
    return {
        "task": metadata['task'],
        "problem": metadata['problem'],
        "n_features": metadata['n_features'],
        "target_classes": metadata['target_classes'],
        "models_available": metadata['models_available']
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    model_name = input_data.model_name or metadata['best_model']
    
    if model_name not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo {model_name} nao encontrado. Modelos disponiveis: {list(sessions.keys())}"
        )
    
    if len(input_data.features) != metadata['n_features']:
        raise HTTPException(
            status_code=400,
            detail=f"Esperado {metadata['n_features']} features, recebido {len(input_data.features)}"
        )
    
    features_array = np.array(input_data.features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    features_float32 = features_scaled.astype(np.float32)
    
    session = sessions[model_name]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    predictions = session.run([output_name], {input_name: features_float32})[0]
    
    if hasattr(predictions[0], '__len__') and len(predictions[0]) == 2:
        probability = float(predictions[0][1])
    else:
        probability = float(predictions[0])
    
    prediction_class = 1 if probability >= 0.5 else 0
    class_label = metadata['target_classes'][prediction_class]
    
    return PredictionOutput(
        model_used=model_name,
        prediction=prediction_class,
        probability=probability,
        class_label=class_label
    )

@app.post("/predict_batch")
def predict_batch(features_list: List[List[float]], model_name: Optional[str] = None):
    model_name = model_name or metadata['best_model']
    
    if model_name not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo {model_name} nao encontrado"
        )
    
    results = []
    for features in features_list:
        if len(features) != metadata['n_features']:
            results.append({
                "error": f"Features invalidas: esperado {metadata['n_features']}, recebido {len(features)}"
            })
            continue
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        features_float32 = features_scaled.astype(np.float32)
        
        session = sessions[model_name]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        predictions = session.run([output_name], {input_name: features_float32})[0]
        
        if hasattr(predictions[0], '__len__') and len(predictions[0]) == 2:
            probability = float(predictions[0][1])
        else:
            probability = float(predictions[0])
        
        prediction_class = 1 if probability >= 0.5 else 0
        
        results.append({
            "prediction": prediction_class,
            "probability": probability,
            "class_label": metadata['target_classes'][prediction_class]
        })
    
    return {
        "model_used": model_name,
        "total_predictions": len(results),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
