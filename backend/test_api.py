"""Pruebas del endpoint con un agente simulado (no gastan crédito de la API)."""
import json
from types import SimpleNamespace
from fastapi.testclient import TestClient
import main

FAKE = {
    "summary": "Los clientes valoran el café, pero se quejan de la demora.",
    "comments_analysis": [
        {"comment": "Café delicioso", "sentiment": "positivo", "topics": ["calidad del producto"]},
        {"comment": "Tardaron mucho", "sentiment": "negativo", "topics": ["tiempo de espera"]},
    ],
    "top_strengths": ["Calidad del producto"], "top_weaknesses": ["Tiempo de espera"],
    "recommendations": ["Reforzar personal en horas pico."],
    "suggested_replies": [{"for_comment": "Tardaron mucho", "reply": "Gracias, ya estamos mejorando."}],
}


class FakeClient:
    class messages:
        @staticmethod
        def create(**kw):
            return SimpleNamespace(content=[SimpleNamespace(text="```json\n" + json.dumps(FAKE) + "\n```")])


main.get_client = lambda: FakeClient()
c = TestClient(main.app)
body = {"business_name": "Café La Esquina", "business_type": "cafetería",
        "comments": ["Café delicioso", "Tardaron mucho"], "generate_replies": True}


def test_health():
    assert c.get("/health").json() == {"status": "ok"}


def test_analyze_ok():
    r = c.post("/analyze", json=body)
    assert r.status_code == 200
    d = r.json()
    assert d["sentiment"] == {"positive": 50, "neutral": 0, "negative": 50}
    assert len(d["suggested_replies"]) == 1


def test_sin_respuestas_si_no_se_piden():
    assert c.post("/analyze", json={**body, "generate_replies": False}).json()["suggested_replies"] == []


def test_comentarios_vacios():
    assert c.post("/analyze", json={**body, "comments": ["  ", ""]}).status_code == 400


def test_demasiados_comentarios():
    assert c.post("/analyze", json={**body, "comments": ["x"] * 101}).status_code == 413


def test_falla_del_agente():
    main.get_client = lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **k: SimpleNamespace(content=[SimpleNamespace(text="no es json")])))
    assert c.post("/analyze", json=body).status_code == 502
