# diagnosis_cnn_pelican_full.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

# -------------------------
# Pelican Search Algorithm
# -------------------------
class PelicanSearch:
    def __init__(self, obj_function, dim, bounds, population_size=6, pa=0.25, alpha=0.01, max_iter=5, verbose=True):
        self.obj_function = obj_function
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose

    def run(self):
        nests = np.random.rand(self.population_size, self.dim)
        for i in range(self.dim):
            low, high = self.bounds[i]
            nests[:, i] = low + nests[:, i] * (high - low)
        fitness = np.array([self.obj_function(nests[i]) for i in range(self.population_size)])

        for iteration in range(self.max_iter):
            new_nests = nests + self.alpha * np.random.randn(self.population_size, self.dim)
            for i in range(self.dim):
                low, high = self.bounds[i]
                new_nests[:, i] = np.clip(new_nests[:, i], low, high)
            new_fitness = np.array([self.obj_function(new_nests[i]) for i in range(self.population_size)])
            for i in range(self.population_size):
                if new_fitness[i] > fitness[i]:
                    nests[i] = new_nests[i]
                    fitness[i] = new_fitness[i]

            # Abandon worst nests
            K = int(self.pa * self.population_size)
            worst_idx = np.argsort(fitness)[:K]
            nests[worst_idx] = np.random.rand(K, self.dim)
            for i in range(self.dim):
                low, high = self.bounds[i]
                nests[worst_idx, i] = low + nests[worst_idx, i] * (high - low)

            if self.verbose:
                print(f"Iteration {iteration+1}/{self.max_iter}, Best fitness: {fitness.max():.4f}")

        best_idx = np.argmax(fitness)
        return nests[best_idx], fitness[best_idx]

# -------------------------
# Decode solution vector to hyperparameters
# -------------------------
def decode_solution(sol_vector):
    lr = float(sol_vector[0])
    dropout = float(sol_vector[1])
    dense = int(round(sol_vector[2]))
    batch = int(round(sol_vector[3]))
    return {"learning_rate": lr, "dropout_rate": dropout, "dense_units": dense, "batch_size": batch, "epochs":3}

# -------------------------
# Build and train CNN
# -------------------------
def train_and_evaluate(hp, quick_run=False, callbacks=None):
    img_size = (224,224)
    train_dir = "train"
    val_dir   = "valid"
    test_dir  = "test"

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_dataset = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                                      batch_size=hp["batch_size"], class_mode='categorical')
    val_dataset = val_datagen.flow_from_directory(val_dir, target_size=img_size,
                                                  batch_size=hp["batch_size"], class_mode='categorical', shuffle=False)
    test_dataset = val_datagen.flow_from_directory(test_dir, target_size=img_size,
                                                   batch_size=hp["batch_size"], class_mode='categorical', shuffle=False)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(hp["dense_units"], activation='relu'),
        layers.Dropout(hp["dropout_rate"]),
        layers.Dense(train_dataset.num_classes, activation='softmax')
    ])
    opt = optimizers.Adam(learning_rate=hp["learning_rate"])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    epochs = hp["epochs"] if not quick_run else 3
    if callbacks is None:
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        callbacks = [early_stop]

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks, verbose=1)
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    return val_acc, model, history, val_dataset, test_dataset

# -------------------------
# Objective function
# -------------------------
def make_objective():
    cache = {}
    def obj_fn(x_vector):
        hp = decode_solution(x_vector)
        key = json.dumps(hp, sort_keys=True)
        if key in cache:
            return cache[key]
        try:
            val_acc, _, _, _, _ = train_and_evaluate(hp, quick_run=True)
        except Exception as e:
            print("Training failed:", hp, e)
            val_acc = 0.0
        cache[key] = val_acc
        print(f"Evaluated {hp} -> val_acc: {val_acc:.4f}")
        return val_acc
    return obj_fn

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    bounds = [(1e-5,1e-3), (0.1,0.6), (64,512), (8,64)]
    dim = len(bounds)
    objective = make_objective()

    ps = PelicanSearch(obj_function=objective, dim=dim, bounds=bounds,
                       population_size=6, pa=0.25, alpha=0.01, max_iter=5, verbose=True)
    best_vec, best_fit = ps.run()

    best_hp = {
        "learning_rate": float(best_vec[0]),
        "dropout_rate": float(best_vec[1]),
        "dense_units": int(round(best_vec[2])),
        "batch_size": int(round(best_vec[3])),
        "epochs": 12
    }
    print("Best hyperparameters for final training:", best_hp)

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    val_acc, final_model, history, val_dataset, test_dataset = train_and_evaluate(best_hp, quick_run=False, callbacks=[early_stop])
    print("Final validation accuracy:", val_acc)

    final_model.save("output/final_pelican_model.h5")
    print("Saved final model to output/final_pelican_model.h5")

    # -------------------------
    # Accuracy & Loss plots
    # -------------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/accuracy_loss_plot.png")
    plt.close()
    print("Saved accuracy & loss plot to output/accuracy_loss_plot.png")

    # -------------------------
    # Confusion matrices & classification reports
    # -------------------------
    def save_confusion(model, dataset, filename, title="Confusion Matrix"):
        y_true, y_pred = [], []
        for images, labels in dataset:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=list(dataset.class_indices.keys()))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.savefig(filename)
        plt.close()
        print(f"Saved {title} to {filename}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(dataset.class_indices.keys())))

    save_confusion(final_model, val_dataset, "output/val_confusion_matrix.png", "Validation Confusion Matrix")
    save_confusion(final_model, test_dataset, "output/test_confusion_matrix.png", "Test Confusion Matrix")
