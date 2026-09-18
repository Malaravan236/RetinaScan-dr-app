import io
import json
import threading

import numpy as np
from PIL import Image
from django.conf import settings

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


_model = None
_model_lock = threading.Lock()

_class_indices = None
_class_lock = threading.Lock()


def get_model():

    global _model

    if _model is None:

        with _model_lock:

            if _model is None:

                import tensorflow as tf

                _model = tf.keras.models.load_model(
                    str(settings.ML_MODEL_PATH)
                )

                print("================================")
                print("MODEL LOADED")
                print("MODEL PATH:", settings.ML_MODEL_PATH)
                print("================================")

    return _model


def get_class_indices():

    global _class_indices

    if _class_indices is None:

        with _class_lock:

            if _class_indices is None:

                model_path = str(
                    settings.ML_MODEL_PATH
                )

                model_dir = (
                    model_path.rsplit("\\", 1)[0]
                    if "\\" in model_path
                    else model_path.rsplit("/", 1)[0]
                )

                class_file = (
                    model_dir +
                    "/class_indices.json"
                )

                try:

                    with open(
                        class_file,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        _class_indices = json.load(file)

                    print(
                        "CLASS MAPPING:",
                        _class_indices
                    )

                except Exception:

                    _class_indices = {
                        "DR": 0,
                        "No DR": 1
                    }

    return _class_indices


def preprocess_image_bytes(
    image_bytes: bytes
) -> np.ndarray:

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # SAME preprocessing as training
    array = preprocess_input(array)

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


def predict_from_bytes(
    image_bytes: bytes
) -> dict:

    model = get_model()

    class_indices = get_class_indices()

    processed_image = preprocess_image_bytes(
        image_bytes
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    index_to_class = {
        int(index): name
        for name, index
        in class_indices.items()
    }

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = index_to_class.get(
        predicted_index,
        "Unknown"
    )

    confidence = float(
        predictions[predicted_index]
    )

    dr_index = class_indices.get("DR")
    no_dr_index = class_indices.get("No DR")

    dr_probability = (
        float(predictions[dr_index])
        if dr_index is not None
        else 0.0
    )

    no_dr_probability = (
        float(predictions[no_dr_index])
        if no_dr_index is not None
        else 0.0
    )

    print("\n================================")
    print("RAW MODEL OUTPUT:", predictions)
    print("CLASS INDICES:", class_indices)
    print("PREDICTED INDEX:", predicted_index)
    print("PREDICTED CLASS:", predicted_class)
    print("CONFIDENCE:", confidence)
    print("DR PROBABILITY:", dr_probability)
    print("NO DR PROBABILITY:", no_dr_probability)
    print("================================")

    return {
        "predicted_class": predicted_class,
        "dr_probability": dr_probability,
        "no_dr_probability": no_dr_probability,
        "confidence": confidence,
        "predicted_index": predicted_index
    }