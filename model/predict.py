# model/predict.py
import sys, importlib.util
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
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
