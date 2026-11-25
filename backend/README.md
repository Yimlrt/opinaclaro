# OpinaClaro - Backend

API en FastAPI que recibe comentarios de clientes y usa un agente de IA (API de Claude) para analizarlos.

## Ejecutar en local
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # y pega tu ANTHROPIC_API_KEY
uvicorn main:app --reload
```
Documentación interactiva: http://localhost:8000/docs

## Probar
```
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"business_name":"Café La Esquina","business_type":"cafetería","comments":["Café delicioso pero tardaron 25 minutos"],"generate_replies":true}'
```
Pruebas automáticas (sin gastar crédito): `pytest -q`

## Despliegue (Render)
Build: `pip install -r requirements.txt` · Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
Variables: ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ALLOWED_ORIGINS (URL del frontend).
