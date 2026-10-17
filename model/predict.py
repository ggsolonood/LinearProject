# model/predict.py
"""
Wrapper for predict.py located at root (Circular Import Safe)
"""
import sys
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ROOT_PREDICT = _ROOT / "predict.py"

main = None
BoxPredictor = None
ModelParameters = None
PredictionOutput = None
load_image = None

if _ROOT_PREDICT.exists():
    _spec = importlib.util.spec_from_file_location("parcel_box_root_predict", str(_ROOT_PREDICT))
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["parcel_box_root_predict"] = _mod
        _spec.loader.exec_module(_mod)
        main = getattr(_mod, "main", None)
        BoxPredictor = getattr(_mod, "BoxPredictor", None)
        ModelParameters = getattr(_mod, "ModelParameters", None)
        PredictionOutput = getattr(_mod, "PredictionOutput", None)
        load_image = getattr(_mod, "load_image", None)

if __name__ == "__main__":
    if main:
        main()
