# model/predict.py
<<<<<<< HEAD
import sys, importlib.util
=======
"""
Wrapper for predict.py located at root (Circular Import Safe)
"""
import sys
import importlib.util
>>>>>>> 81a42b11c0faa0e5d08b03043fe00569ffc6796b
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
<<<<<<< HEAD
_P = _ROOT / 'predict.py'
main = BoxPredictor = ModelParameters = PredictionOutput = load_image = None
if _P.exists():
    _s = importlib.util.spec_from_file_location('parcel_box_root_predict', str(_P))
    if _s and _s.loader:
        _m = importlib.util.module_from_spec(_s)
        sys.modules['parcel_box_root_predict'] = _m
        _s.loader.exec_module(_m)
        main = getattr(_m, 'main', None)
        BoxPredictor = getattr(_m, 'BoxPredictor', None)
        ModelParameters = getattr(_m, 'ModelParameters', None)
        PredictionOutput = getattr(_m, 'PredictionOutput', None)
        load_image = getattr(_m, 'load_image', None)
if __name__ == '__main__':
    if main: main()
=======
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
>>>>>>> 81a42b11c0faa0e5d08b03043fe00569ffc6796b
