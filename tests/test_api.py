from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_setosa():
    r = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 200
    assert r.json()["class_name"] == "setosa"


def test_predict_invalid_input():
    r = client.post("/predict", json={"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 422


def test_monitoring():
    client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    r = client.get("/monitoring")
    assert r.status_code == 200
    body = r.json()
    assert body["total_predictions"] >= 1
    assert len(body["derive_par_variable"]) == 4


def test_home_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "Iris" in r.text
