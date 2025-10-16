# bird/inference.py
from django.conf import settings
import os, json, numpy as np
from PIL import Image

# librosa for audio
try:
    import librosa
except Exception:
    librosa = None

# support both tf.keras and keras 3
try:
    from tensorflow.keras.models import load_model as tf_load_model
except Exception:
    tf_load_model = None
try:
    from keras.models import load_model as k_load_model
except Exception:
    k_load_model = None

# Optional: rich metadata (safe fallback if not present)
try:
    from .species_meta import SPECIES
except Exception:
    SPECIES = {}

_image_model = None
_audio_model = None


# ---------- helpers ----------
def _load_model(path):
    path = str(path)
    loaders = [p for p in (tf_load_model, k_load_model) if p is not None]
    last = None
    for L in loaders:
        try:
            return L(path, compile=False)
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to load model at {path}: {last}")

def _maybe_load_index_json(json_path):
    if not json_path:
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {int(k): v for k, v in data.items()}
        if isinstance(data, list):
            return {int(d.get("index")): d.get("label") for d in data}
    except Exception:
        return None
    return None

def _align_classnames(default_list, index_json_path):
    idxmap = _maybe_load_index_json(index_json_path)
    if idxmap:
        return [idxmap[i] for i in sorted(idxmap)]
    return default_list

def _softmax(x):
    x = np.array(x, dtype="float32").reshape(-1)
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-8)

def _topk(probs, class_names, k=5):
    probs = np.array(probs, dtype="float32").reshape(-1)
    if probs.shape[-1] != len(class_names):
        raise ValueError(
            f"Output dim {probs.shape[-1]} != class list {len(class_names)}. "
            f"Fix AUDIO_CLASS_NAMES / RAW_CLASS_NAMES or set *_CLASS_INDEX_JSON."
        )
    idx = np.argsort(-probs)[:min(k, len(class_names))]
    return [(int(i), class_names[int(i)], float(probs[int(i)])) for i in idx]

def _meta_for(label):
    return SPECIES.get(label, {
        "display_name": label,
        "scientific_name": label,
        "habitat": "-",
        "facts": []
    })


# ---------- model loaders ----------
def load_image_model():
    global _image_model
    if _image_model is None:
        _image_model = _load_model(getattr(settings, "IMAGE_MODEL_PATH"))
    return _image_model

def load_audio_model():
    global _audio_model
    if _audio_model is None:
        _audio_model = _load_model(getattr(settings, "AUDIO_MODEL_PATH"))
    return _audio_model


# ---------- image preprocessing ----------
def _img_target_from_model(model):
    try:
        ishape = model.input_shape  # e.g., (None, 224, 224, 3) or [(...), (...)]
        if isinstance(ishape, (list, tuple)):
            ishape = ishape[0]  # first input is usually the image
        if len(ishape) == 4:
            # NHWC vs NCHW
            if ishape[1] in (1, 3):
                return int(ishape[2]), int(ishape[3]), "NCHW"
            return int(ishape[1]), int(ishape[2]), "NHWC"
    except Exception:
        pass
    h, w = getattr(settings, "IMAGE_TARGET_SIZE", (224, 224))
    return h, w, "NHWC"

def _prep_img_variants(img_path, model):
    from PIL import Image
    import numpy as np

    H, W, layout = _img_target_from_model(model)
    img = Image.open(img_path).convert("RGB").resize((W, H))
    x = np.array(img).astype("float32") / 255.0

    # Convert to correct layout
    if layout == "NHWC":
        x = x[None, ...]
    else:
        x = np.transpose(x, (2, 0, 1))[None, ...]

    # If model expects 2 inputs, feed both as same image
    inputs = getattr(model, "inputs", None)
    if inputs and len(inputs) == 2:
        return [[x, x]]
    else:
        return [x]



def _pick_best_distribution(outputs):
    # choose the output that has the highest peak probability
    best = None
    best_peak = -1.0
    for out in outputs:
        out = np.array(out).reshape(-1)
        probs = _softmax(out)
        peak = float(np.max(probs))
        if peak > best_peak:
            best_peak, best = peak, probs
    return best


# ---------- public: image prediction ----------
def predict_image(img_path, class_names):
    model = load_image_model()
    class_names = _align_classnames(
        class_names,
        getattr(settings, "IMAGE_CLASS_INDEX_JSON", None)
    )

    variants = _prep_img_variants(img_path, model)
    outs = []
    for v in variants:
        if isinstance(v, list):  # multi-input model
            outs.append(model.predict(v, verbose=0)[0])
        else:
            outs.append(model.predict(v, verbose=0)[0])

    probs = _pick_best_distribution(outs)
    top5 = _topk(probs, class_names, k=5)
    best_idx, best_label, best_p = top5[0]
    return {
        "label": best_label,
        "score": float(best_p),
        "meta": _meta_for(best_label),
        "top5": [(name, float(p)) for _, name, p in top5],
    }


# ---------- audio preprocessing & prediction ----------
def _preprocess_audio(path):
    if librosa is None:
        raise RuntimeError("librosa not installed")
    y, sr = librosa.load(path, sr=22050, mono=True)
    D = librosa.stft(y, n_fft=512, hop_length=256, win_length=512, center=True)
    S = np.log1p(np.abs(D)).T  # (frames, 257)

    T, F = getattr(settings, "AUDIO_TARGET_SHAPE", (1026, 257))
    # pad/crop freq
    if S.shape[1] > F: S = S[:, :F]
    if S.shape[1] < F: S = np.hstack([S, np.zeros((S.shape[0], F - S.shape[1]))])
    # pad/crop time
    if S.shape[0] > T: S = S[:T, :]
    if S.shape[0] < T: S = np.vstack([S, np.zeros((T - S.shape[0], S.shape[1]))])

    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    return S  # (T,F)

def predict_audio(audio_path, class_names):
    model = load_audio_model()
    class_names = _align_classnames(
        class_names,
        getattr(settings, "AUDIO_CLASS_INDEX_JSON", None)
    )

    X = _preprocess_audio(audio_path)  # (T,F)
    # model may expect (B,T,F) or (B,T,F,1)
    ishape = model.input_shape
    if isinstance(ishape, (list, tuple)):
        ishape = ishape[0]
    want_rank = len(ishape)

    if want_rank == 3:
        X = X[None, ...].astype("float32")
    else:
        X = X[..., None][None, ...].astype("float32")

    raw = model.predict(X, verbose=0)[0]
    probs = _softmax(raw)
    top5 = _topk(probs, class_names, k=5)
    best_idx, best_label, best_p = top5[0]
    return {
        "label": best_label,
        "score": float(best_p),
        "meta": _meta_for(best_label),
        "top5": [(name, float(p)) for _, name, p in top5],
    }
