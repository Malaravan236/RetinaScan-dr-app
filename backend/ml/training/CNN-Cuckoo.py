import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = r"C:\Users\HP\Downloads\Diagnosis of Diabetic Retinopathy (3)\Diagnosis of Diabetic Retinopathy\src\train"
TEST_DIR = r"C:\Users\HP\Downloads\Diagnosis of Diabetic Retinopathy (3)\Diagnosis of Diabetic Retinopathy\src\test"
OUTPUT_DIR = os.path.join(BASE_DIR, 'final_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
print("📂 Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -----------------------------
# CNN MODEL FUNCTION
# -----------------------------
def create_cnn(learning_rate=0.001, dropout_rate=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# -----------------------------
# CUCKOO SEARCH IMPLEMENTATION
# -----------------------------
class CuckooSearch:
    def __init__(self, n_nests, pa, beta, obj_func, bounds):
        self.n_nests = n_nests
        self.pa = pa
        self.beta = beta
        self.obj_func = obj_func
        self.bounds = bounds

    def levy_flight(self, lam):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
                 (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
        u = np.random.normal(0, sigma, len(self.bounds))
        v = np.random.normal(0, 1, len(self.bounds))
        step = u / (np.abs(v) ** (1 / lam))
        return step

    def run(self, n_iter=5):
        nests = np.random.rand(self.n_nests, len(self.bounds))
        for i in range(len(self.bounds)):
            nests[:, i] = nests[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]

        fitness = np.array([self.obj_func(n) for n in nests])
        best_nest = nests[np.argmax(fitness)]
        best_fitness = np.max(fitness)

        print("🔍 Running Cuckoo Search Optimization...")

        for _ in range(n_iter):
            new_nests = nests + 0.01 * np.random.randn(*nests.shape) * self.levy_flight(self.beta)
            for i in range(len(self.bounds)):
                new_nests[:, i] = np.clip(new_nests[:, i], self.bounds[i][0], self.bounds[i][1])
            new_fitness = np.array([self.obj_func(n) for n in new_nests])

            for i in range(self.n_nests):
                if new_fitness[i] > fitness[i]:
                    fitness[i] = new_fitness[i]
                    nests[i] = new_nests[i]

            k = np.random.rand(self.n_nests) < self.pa
            nests[k] = np.random.rand(np.sum(k), len(self.bounds))
            for i in range(len(self.bounds)):
                nests[k, i] = nests[k, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]

            idx = np.argmax(fitness)
            if fitness[idx] > best_fitness:
                best_fitness = fitness[idx]
                best_nest = nests[idx]

        return best_nest

# -----------------------------
# OBJECTIVE FUNCTION
# -----------------------------
def objective(params):
    lr, dr = params
    model = create_cnn(learning_rate=lr, dropout_rate=dr)
    history = model.fit(train_generator, epochs=2, validation_data=val_generator, verbose=0)
    val_acc = np.max(history.history['val_accuracy'])
    return val_acc

# -----------------------------
# RUN CUCKOO SEARCH
# -----------------------------
cs = CuckooSearch(
    n_nests=5,
    pa=0.25,
    beta=1.5,
    obj_func=objective,
    bounds=[(1e-4, 1e-2), (0.3, 0.7)]  # [learning_rate, dropout_rate]
)

best_vec = cs.run()
best_lr, best_dr = best_vec
print(f"✅ Best Hyperparameters: LR={best_lr:.5f}, Dropout={best_dr:.2f}")

# -----------------------------
# FINAL MODEL TRAINING
# -----------------------------
print("🚀 Training final CNN model with optimized parameters...")

model = create_cnn(learning_rate=best_lr, dropout_rate=best_dr)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# -----------------------------
# EVALUATE MODEL
# -----------------------------
print("🔍 Evaluating model on test data...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# -----------------------------
# PLOT ACCURACY & LOSS
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
acc_loss_path = os.path.join(OUTPUT_DIR, f'cuckoo_accuracy_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
plt.savefig(acc_loss_path)
plt.show()
print(f"📈 Accuracy & Loss graph saved at {acc_loss_path}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
cm_path = os.path.join(OUTPUT_DIR, f'cuckoo_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
plt.savefig(cm_path)
plt.show()
print(f"📊 Confusion matrix saved at {cm_path}")

# -----------------------------
# SAVE MODEL
# -----------------------------
model_path = os.path.join(OUTPUT_DIR, f'cuckoo_cnn_dr_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
model.save(model_path)
print(f"💾 Model saved at {model_path}")
