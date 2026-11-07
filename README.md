# RupeeLens V2

AI-powered Pakistani banknote denomination recognition prototype, built with Streamlit and YOLO11n classification.

**Supported values:** Rs. 10, 20, 50, 100, 500, 1,000.

## Deployment

- Streamlit Community Cloud entry point: `streamlit_app.py`
- Python version: 3.11
- Dependencies: `requirements.txt` and `packages.txt`
- Model checkpoint required at `runs/classifier_smoke/weights/best.pt` (not included until uploaded from the owner's PC).

This is an experimental **single-image classification** demonstration, not a counterfeit detector, currency authentication system, or multi-banknote counter. Its confidence score is not proof of a banknote's authenticity or correctness.

Upload images only if you are comfortable processing them on a hosted service.
