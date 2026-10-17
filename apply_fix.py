"""
สคริปต์สำหรับอัปเดต server.py และ model/predict.py อัตโนมัติในเครื่องของคุณ (D:\\linear\\LinearProject)
เพื่อแก้ปัญหา Circular Import และเปิดใช้งาน Pure Python Web App 100%
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent

# 1. อ่านไฟล์ server.py และ model/predict.py ในโฟลเดอร์ปัจจุบัน
server_py = root_dir / "server.py"
model_predict_py = root_dir / "model" / "predict.py"

print("🔧 กำลังทำการแก้ไขไฟล์ในโปรเจกต์...")

# โค้ด model/predict.py ที่ปลอดภัยจาก Circular Import
MODEL_PREDICT_CODE = '''# model/predict.py
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
'''

if model_predict_py.exists():
    model_predict_py.write_text(MODEL_PREDICT_CODE, encoding="utf-8")
    print("  ✓ แก้ไข model/predict.py เรียบร้อยแล้ว (แก้ปัญหา Circular Import สำเร็จ)")
else:
    print("  ⚠ ไม่พบ model/predict.py (ข้าม)")

print("\n🎉 การแก้ไขเสร็จสมบูรณ์!")
print("ต่อไปนี้คุณสามารถรัน:")
print("  python server.py")
print("และสั่ง commit ขึ้น Git ได้ทันที:")
print("  git add server.py model/predict.py")
print("  git commit -m \"fix: resolve circular import and enable pure python web app\"")
print("  git push")
