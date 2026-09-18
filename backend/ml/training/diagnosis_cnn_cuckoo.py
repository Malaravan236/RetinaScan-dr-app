# diagnosis_cnn_cuckoo_full_finetune.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CUCKOO SEARCH
# ============================================================

class CuckooSearch:

    def __init__(
        self,
        objective_function,
        bounds,
        population_size=6,
        pa=0.25,
        alpha=0.01,
        max_iter=5
    ):
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.max_iter = max_iter

    def random_solution(self):
        return np.array([
            np.random.uniform(low, high)
            for low, high in self.bounds
        ])

    def levy_flight(self, solution):
        step = np.random.normal(
            loc=0.0,
            scale=self.alpha,
            size=len(solution)
        )

        new_solution = solution + step

        for i, (low, high) in enumerate(self.bounds):
            new_solution[i] = np.clip(
                new_solution[i],
                low,
                high
            )

        return new_solution

    def run(self):

        nests = np.array([
            self.random_solution()
            for _ in range(self.population_size)
        ])

        fitness = np.array([
            self.objective_function(solution)
            for solution in nests
        ])

        best_index = np.argmax(fitness)
        best_solution = nests[best_index].copy()
        best_fitness = fitness[best_index]

        print("\n========================================")
        print("CUCKOO SEARCH STARTED")
        print("========================================")

        for iteration in range(self.max_iter):

            print(
                f"\nCuckoo Search Iteration "
                f"{iteration + 1}/{self.max_iter}"
            )

            for i in range(self.population_size):

                new_solution = self.levy_flight(nests[i])

                new_fitness = self.objective_function(
                    new_solution
                )

                random_index = np.random.randint(
                    0,
                    self.population_size
                )

                if new_fitness > fitness[random_index]:

                    nests[random_index] = new_solution
                    fitness[random_index] = new_fitness

                    if new_fitness > best_fitness:

                        best_fitness = new_fitness
                        best_solution = new_solution.copy()

                        print(
                            "New best fitness:",
                            best_fitness
                        )

            # Abandon some nests
            for i in range(self.population_size):

                if np.random.rand() < self.pa:

                    nests[i] = self.random_solution()

                    fitness[i] = self.objective_function(
                        nests[i]
                    )

                    if fitness[i] > best_fitness:

                        best_fitness = fitness[i]
                        best_solution = nests[i].copy()

            print(
                "Best fitness:",
                best_fitness
            )

        print("\n========================================")
        print("CUCKOO SEARCH FINISHED")
        print("========================================")

        return best_solution, best_fitness


# ============================================================
# 2. DECODE HYPERPARAMETERS
# ============================================================

def decode_solution(sol_vector):

    learning_rate = float(sol_vector[0])

    dropout_rate = float(sol_vector[1])

    dense_units = int(
        round(sol_vector[2])
    )

    batch_size = int(
        round(sol_vector[3])
    )

    return {
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "batch_size": batch_size,
        "epochs": 3
    }


# ============================================================
# 3. TRAIN AND EVALUATE
# ============================================================

def train_and_evaluate(
    hp,
    quick_run=False,
    callbacks=None,
    fine_tune=False
):

    img_size = (224, 224)

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    train_dir = "train"
    val_dir = "valid"
    test_dir = "test"

    # Check folders
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training folder not found: {os.path.abspath(train_dir)}"
        )

    if not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Validation folder not found: {os.path.abspath(val_dir)}"
        )

    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Testing folder not found: {os.path.abspath(test_dir)}"
        )

    # --------------------------------------------------------
    # Data generators
    # --------------------------------------------------------

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=hp["batch_size"],
        class_mode="categorical",
        shuffle=True
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=hp["batch_size"],
        class_mode="categorical",
        shuffle=False
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = val_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=hp["batch_size"],
        class_mode="categorical",
        shuffle=False
    )

    # ========================================================
    # IMPORTANT: CLASS MAPPING
    # ========================================================

    print("\n========================================")
    print("CLASS MAPPING")
    print("========================================")

    print(
        "TRAIN CLASS INDICES:",
        train_dataset.class_indices
    )

    print(
        "VALID CLASS INDICES:",
        val_dataset.class_indices
    )

    print(
        "TEST CLASS INDICES:",
        test_dataset.class_indices
    )

    # Make sure all datasets have same class mapping
    if train_dataset.class_indices != val_dataset.class_indices:
        raise ValueError(
            "TRAIN and VALID class mapping are different!"
        )

    if train_dataset.class_indices != test_dataset.class_indices:
        raise ValueError(
            "TRAIN and TEST class mapping are different!"
        )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n========================================")
    print("CREATING MOBILENETV2 MODEL")
    print("========================================")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model initially
    base_model.trainable = False

    model = models.Sequential([
        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            hp["dense_units"],
            activation="relu"
        ),

        layers.Dropout(
            hp["dropout_rate"]
        ),

        layers.Dense(
            train_dataset.num_classes,
            activation="softmax"
        )
    ])

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optimizers.Adam(
        learning_rate=hp["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel created successfully.")

    # ========================================================
    # CALLBACK
    # ========================================================

    if callbacks is None:

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        callbacks = [early_stop]

    # ========================================================
    # TRAINING
    # ========================================================

    epochs = (
        hp["epochs"]
        if not quick_run
        else 3
    )

    print("\n========================================")
    print("INITIAL TRAINING")
    print("========================================")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # ========================================================
    # FINE TUNING
    # ========================================================

    if fine_tune:

        print("\n========================================")
        print("FINE TUNING MOBILENETV2")
        print("========================================")

        base_model.trainable = True

        # Freeze all layers except last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Keep BatchNormalization layers frozen
        for layer in base_model.layers:

            if isinstance(
                layer,
                layers.BatchNormalization
            ):
                layer.trainable = False

        fine_tune_optimizer = optimizers.Adam(
            learning_rate=hp["learning_rate"] / 10
        )

        model.compile(
            optimizer=fine_tune_optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        fine_tune_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=5,
            callbacks=callbacks,
            verbose=1
        )

        # Combine histories
        for key in history.history.keys():

            if key in fine_tune_history.history:

                history.history[key].extend(
                    fine_tune_history.history[key]
                )

    # ========================================================
    # VALIDATION EVALUATION
    # ========================================================

    print("\n========================================")
    print("VALIDATION RESULT")
    print("========================================")

    val_loss, val_acc = model.evaluate(
        val_dataset,
        verbose=0
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_acc:.4f}"
    )

    return (
        val_acc,
        model,
        history,
        val_dataset,
        test_dataset
    )


# ============================================================
# 4. CUCKOO OBJECTIVE FUNCTION
# ============================================================

def make_objective():

    cache = {}

    def objective_function(x_vector):

        hp = decode_solution(x_vector)

        key = tuple(
            round(float(x), 6)
            for x in x_vector
        )

        if key in cache:

            print(
                "Using cached result:",
                hp
            )

            return cache[key]

        print("\n----------------------------------------")
        print("Testing Hyperparameters")
        print("----------------------------------------")

        print(
            "Learning Rate:",
            hp["learning_rate"]
        )

        print(
            "Dropout:",
            hp["dropout_rate"]
        )

        print(
            "Dense Units:",
            hp["dense_units"]
        )

        print(
            "Batch Size:",
            hp["batch_size"]
        )

        try:

            val_acc, _, _, _, _ = train_and_evaluate(
                hp,
                quick_run=True
            )

        except Exception as e:

            print(
                "Training error:",
                str(e)
            )

            val_acc = 0.0

        cache[key] = val_acc

        return val_acc

    return objective_function


# ============================================================
# 5. PLOT TRAINING HISTORY
# ============================================================

def plot_training_history(history):

    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "output/training_accuracy.png"
    )

    plt.show()

    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "output/training_loss.png"
    )

    plt.show()

    plt.close()


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

def plot_confusion(
    model,
    dataset,
    output_path="output/confusion_matrix.png"
):

    print("\n========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    # Make sure generator starts from beginning
    dataset.reset()

    # Predict all images
    predictions = model.predict(
        dataset,
        verbose=1
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    y_true = dataset.classes

    class_names = [
        name
        for name, index
        in sorted(
            dataset.class_indices.items(),
            key=lambda x: x[1]
        )
    ]

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nClass Names:")

    for index, name in enumerate(class_names):

        print(
            f"{index} = {name}"
        )

    print("\nConfusion Matrix:")

    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    disp.plot(
        ax=ax,
        values_format="d"
    )

    plt.title(
        "Diabetic Retinopathy Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path
    )

    plt.show()

    plt.close()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print(report)

    return cm, report


# ============================================================
# 7. SAVE CLASS MAPPING
# ============================================================

def save_class_mapping(
    class_indices,
    output_path="output/class_indices.json"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_indices,
            file,
            indent=4
        )

    print(
        "\nClass mapping saved to:",
        output_path
    )


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    # Create output folder
    os.makedirs(
        "output",
        exist_ok=True
    )

    print("\n")
    print("========================================")
    print("DIABETIC RETINOPATHY")
    print("MOBILENETV2 + CUCKOO SEARCH")
    print("========================================")

    # ========================================================
    # HYPERPARAMETER BOUNDS
    # ========================================================

    bounds = [
        (1e-5, 1e-3),   # Learning rate
        (0.1, 0.6),     # Dropout
        (64, 512),      # Dense units
        (8, 64)         # Batch size
    ]

    # ========================================================
    # OBJECTIVE
    # ========================================================

    objective = make_objective()

    # ========================================================
    # CUCKOO SEARCH
    # ========================================================

    cuckoo = CuckooSearch(
        objective_function=objective,
        bounds=bounds,
        population_size=6,
        pa=0.25,
        alpha=0.01,
        max_iter=5
    )

    best_vector, best_fitness = cuckoo.run()

    # ========================================================
    # BEST HYPERPARAMETERS
    # ========================================================

    best_hp = {
        "learning_rate": float(
            best_vector[0]
        ),

        "dropout_rate": float(
            best_vector[1]
        ),

        "dense_units": int(
            round(best_vector[2])
        ),

        "batch_size": int(
            round(best_vector[3])
        ),

        "epochs": 12
    }

    print("\n========================================")
    print("BEST HYPERPARAMETERS")
    print("========================================")

    print(
        json.dumps(
            best_hp,
            indent=4
        )
    )

    print(
        "\nBest Validation Fitness:",
        best_fitness
    )

    # ========================================================
    # FINAL TRAINING
    # ========================================================

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    (
        val_acc,
        final_model,
        history,
        val_dataset,
        test_dataset
    ) = train_and_evaluate(
        best_hp,
        quick_run=False,
        callbacks=[early_stop],
        fine_tune=True
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        "output/final_cuckoo_model.h5"
    )

    final_model.save(
        model_path
    )

    print("\n========================================")
    print("MODEL SAVED")
    print("========================================")

    print(
        os.path.abspath(model_path)
    )

    # ========================================================
    # SAVE CLASS MAPPING
    # ========================================================

    save_class_mapping(
        train_class_indices := val_dataset.class_indices
    )

    # ========================================================
    # TEST DATASET EVALUATION
    # ========================================================

    print("\n========================================")
    print("TEST DATASET RESULT")
    print("========================================")

    test_loss, test_accuracy = final_model.evaluate(
        test_dataset,
        verbose=1
    )

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    # ========================================================
    # TRAINING GRAPHS
    # ========================================================

    plot_training_history(
        history
    )

    # ========================================================
    # CONFUSION MATRIX + REPORT
    # ========================================================

    plot_confusion(
        final_model,
        test_dataset
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n========================================")
    print("TRAINING COMPLETED")
    print("========================================")

    print(
        "Validation Accuracy:",
        val_acc
    )

    print(
        "Test Accuracy:",
        test_accuracy
    )

    print(
        "Model:",
        model_path
    )

    print(
        "Class Mapping:",
        val_dataset.class_indices
    )

    print("========================================")