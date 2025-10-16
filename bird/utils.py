# bird/utils.py
from django.conf import settings
import re

# If you have the 205 list, import it; else use placeholders
try:
    from .species_meta import RAW_CLASS_NAMES, SPECIES, HOME_SHOWCASE_20
except Exception:
    RAW_CLASS_NAMES = [f"class_{i}" for i in range(205)]
    SPECIES = {}
    HOME_SHOWCASE_20 = RAW_CLASS_NAMES[:20]

# Public class lists
IMAGE_CLASS_NAMES = RAW_CLASS_NAMES
AUDIO_CLASS_NAMES = getattr(settings, "AUDIO_CLASS_NAMES", [])

SPECIES_META = SPECIES

def _slug_from_key(key: str) -> str:
    s = key.lower().replace('.', '-').replace('_', '-')
    s = re.sub(r'[^a-z0-9\-]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s

IMG_EXT = getattr(settings, "SPECIES_IMAGE_EXT", "jpg")

SHOWCASE = []
for k in HOME_SHOWCASE_20:
    item = dict(SPECIES_META.get(k, {"display_name": k}))
    item["image_file"] = f"{_slug_from_key(k)}.{IMG_EXT}"
    SHOWCASE.append(item)
