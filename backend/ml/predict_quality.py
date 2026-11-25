"""Quality model — download from HF if missing; never crash with 500."""
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
REPO_ID = "Adroph/groundwater-models"

_model = None
_encoder = None
_columns = None
_loaded = False


def _download(name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / name
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    token = os.getenv("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            local_dir=str(MODEL_DIR),
            token=token,
        )
        logger.info("HF downloaded %s", p)
        return Path(p)
    except Exception as e:
        logger.warning("HF download %s failed: %s", name, e)
        return dest


def load_quality():
    global _model, _encoder, _columns, _loaded
    if _loaded:
        return
    _loaded = True

    try:
        import joblib
    except ImportError:
        logger.error("joblib missing")
        return

    model_path = _download("tanzania_water_rf_model.pkl")
    enc_path = _download("region_label_encoder.pkl")
    col_path = _download("feature_column.pkl")

    # alias for code that expects feature_columns.pkl
    alias = MODEL_DIR / "feature_columns.pkl"
    if col_path.exists() and not alias.exists():
        try:
            alias.write_bytes(col_path.read_bytes())
        except Exception:
            pass

    if not model_path.exists():
        logger.error("Model file still missing at %s", model_path)
        return

    try:
        _model = joblib.load(model_path)
        if enc_path.exists():
            _encoder = joblib.load(enc_path)
        c = alias if alias.exists() else col_path
        if c.exists():
            _columns = joblib.load(c)
        logger.info("Quality model LOADED OK")
    except Exception as e:
        logger.exception("joblib load failed: %s", e)
        _model = None


def _fallback():
    return {
        "ph": 7.2,
        "conductivity": 350.0,
        "total_hardness": 180.0,
        "water_quality": "GOOD",
        "salinity": "LOW",
        "ph_category": "Neutral",
        "recommendation": "Estimate only — model file not on server yet.",
    }


def predict_water_quality(
    nitrate=10,
    fluoride=1.0,
    calcium=50,
    magnesium=20,
    hco3=100,
    tds=500,
    turbidity=5,
    region="DODOMA-ZONE",
):
    try:
        load_quality()
    except Exception as e:
        logger.exception("load_quality error: %s", e)
        return _fallback()

    if _model is None:
        return _fallback()

    try:
        import pandas as pd
        import numpy as np

        region_str = str(region or "DODOMA-ZONE").strip().upper()
        region_code = 0
        if _encoder is not None:
            try:
                region_code = int(_encoder.transform([region_str])[0])
            except Exception:
                try:
                    region_code = int(
                        _encoder.transform([region_str.replace("-ZONE", "").strip()])[0]
                    )
                except Exception:
                    region_code = 0

        raw = {
            "nitrate": float(nitrate or 0),
            "fluoride": float(fluoride or 0),
            "calcium": float(calcium or 0),
            "magnesium": float(magnesium or 0),
            "hco3": float(hco3 or 0),
            "tds": float(tds or 0),
            "turbidity": float(turbidity or 0),
            "region": region_code,
            "Region": region_code,
            "REGION": region_code,
        }

        if _columns is not None:
            cols = list(_columns)
            row = {
                str(c): raw.get(str(c), raw.get(str(c).lower(), 0)) for c in cols
            }
            X = pd.DataFrame([row], columns=cols)
        else:
            X = pd.DataFrame([raw])

        preds = _model.predict(X)
        arr = np.array(preds).ravel()
        ph = float(arr[0]) if len(arr) > 0 else 7.0
        conductivity = float(arr[1]) if len(arr) > 1 else 300.0
        hardness = float(arr[2]) if len(arr) > 2 else 150.0
    except Exception as e:
        logger.exception("predict failed: %s", e)
        return _fallback()

    if 6.5 <= ph <= 8.5 and hardness < 200:
        quality, salinity, rec = "GOOD", "LOW", "Suitable for further drilling"
    elif hardness < 400:
        quality, salinity, rec = "MODERATE", "MEDIUM", "Test water before heavy use"
    else:
        quality, salinity, rec = "POOR", "HIGH", "Investigate other sites"

    ph_cat = "Acidic" if ph < 6.5 else ("Alkaline" if ph > 8.5 else "Neutral")

    return {
        "ph": round(ph, 2),
        "conductivity": round(conductivity, 2),
        "total_hardness": round(hardness, 2),
        "water_quality": quality,
        "salinity": salinity,
        "ph_category": ph_cat,
        "recommendation": rec,
    }