import json
from pathlib import Path
from models.schemas import InputRecord
from utils.logger import logger


def load_inputs(filepath: str) -> list[InputRecord]:
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Input file not found: {filepath}")
        raise FileNotFoundError(f"Input file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Input file must contain a JSON array.")

    records = []
    for i, item in enumerate(raw_data):
        try:
            record = InputRecord(**item)
            records.append(record)
            logger.debug(f"Loaded: {record.id}")
        except Exception as e:
            logger.warning(f"Skipping malformed record at index {i}: {e}")

    logger.info(f"Loaded {len(records)} records from {filepath}")
    return records
