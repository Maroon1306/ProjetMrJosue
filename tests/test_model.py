import json

import yaml


def test_model_accuracy_threshold():
    """Le pipeline CI échoue si le modèle passe sous 90 % de précision."""
    params = yaml.safe_load(open("params.yaml"))
    metrics = json.load(open(params["train"]["metrics_path"]))
    assert metrics["accuracy"] >= 0.9
