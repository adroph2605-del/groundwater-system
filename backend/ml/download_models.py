"""Download model .pkl files from Hugging Face into backend/ml/model/"""
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "model"
REPO_ID = "Adroph/groundwater-models"

# Faili zinazohitajika (jina kwenye HF = jina la local)
FILES = [
    "tanzania_water_rf_model.pkl",
    "region_label_encoder.pkl",
    "feature_column.pkl",       # jina kwenye HF
    "rock_names.pkl",
    "rock_type_features.pkl",
    # kubwa — optional kama disk inatosha:
    # "rock_type_model.pkl",
    # "geological_prediction_model.pkl",
]


def download_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed")
        return False

    token = os.getenv("HF_TOKEN")  # required if repo is private

    ok = True
    for name in FILES:
        dest = MODEL_DIR / name
        # feature_columns.pkl = copy ya feature_column.pkl
        if name == "feature_column.pkl":
            dest_alt = MODEL_DIR / "feature_columns.pkl"
        else:
            dest_alt = None

        if dest.exists() and dest.stat().st_size > 0:
            logger.info("Already present: %s", dest)
            if dest_alt and not dest_alt.exists():
                dest_alt.write_bytes(dest.read_bytes())
            continue

        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=name,
                local_dir=str(MODEL_DIR),
                token=token,
            )
            logger.info("Downloaded: %s", path)
            if dest_alt is not None:
                p = Path(path)
                dest_alt.write_bytes(p.read_bytes())
                logger.info("Also saved as feature_columns.pkl")
        except Exception as e:
            logger.exception("Failed %s: %s", name, e)
            ok = False

    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_models()