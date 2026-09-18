# # import tensorflow as tf
# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import os
# # from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# # import seaborn as sns

# # # -------------------------
# # # Load trained model
# # # -------------------------
# # model_path = r"C:\Users\HP\Downloads\Diagnosis of Diabetic Retinopathy (3)\Diagnosis of Diabetic Retinopathy\src\outputs\final_pelican_model.h5"
# # model = tf.keras.models.load_model(model_path)
# # print("✅ Model loaded successfully!")

# # # -------------------------
# # # Prepare test data
# # # -------------------------
# # test_dir = r"C:\Users\HP\Downloads\Diagnosis of Diabetic Retinopathy (3)\Diagnosis of Diabetic Retinopathy\src\test"

# # # ImageDataGenerator for test set
# # test_datagen = ImageDataGenerator(rescale=1./255)

# # test_generator = test_datagen.flow_from_directory(
# #     test_dir,
# #     target_size=(224, 224),
# #     batch_size=32,
# #     class_mode='categorical',
# #     shuffle=False
# # )

# # # -------------------------
# # # Predict on test data
# # # -------------------------
# # preds = model.predict(test_generator, verbose=1)
# # y_pred = np.argmax(preds, axis=1)
# # y_true = test_generator.classes

# # class_labels = list(test_generator.class_indices.keys())

# # # -------------------------
# # # Confusion Matrix
# # # -------------------------
# # cm = confusion_matrix(y_true, y_pred)
# # plt.figure(figsize=(6,5))
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #             xticklabels=class_labels,
# #             yticklabels=class_labels)
# # plt.xlabel('Predicted')
# # plt.ylabel('Actual')
# # plt.title('Confusion Matrix')
# # plt.show()

# # # -------------------------
# # # Accuracy and Classification Report
# # # -------------------------
# # acc = accuracy_score(y_true, y_pred)
# # print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")

# # print("Classification Report:\n")
# # print(classification_report(y_true, y_pred, target_names=class_labels))



import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pickle
import os

# -------------------------
# Load trained model
# -------------------------
model_path = r"C:\Users\HP\Downloads\Diagnosis of Diabetic Retinopathy (3)\Diagnosis of Diabetic Retinopathy\src\outputs\final_cuckoo_model.h5"
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# -------------------------
# Prepare test data
# -------------------------
test_dir = r"C:\Users\HP\Downloads\Diagnosis of Diabetic Retinopathy (3)\Diagnosis of Diabetic Retinopathy\src\test"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# -------------------------
# Predict on test data
# -------------------------
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -------------------------
# Accuracy and Classification Report
# -------------------------
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# -------------------------
# Accuracy & Loss Curves
# -------------------------
# Load history if saved as a pickle file
history_path = r"C:\Users\HP\Downloads\Diagnosis of Diabetic Retinopathy (3)\Diagnosis of Diabetic Retinopathy\src\outputs\history.pkl"
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot training & validation loss
    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("⚠️ History file not found. Accuracy & loss curves cannot be plotted.")



# import matplotlib.pyplot as plt

# # -------------------------
# # Example history dictionary
# # Replace these lists with your actual history.history data
# # -------------------------
# history = {
#     'accuracy': [0.85, 0.88, 0.90, 0.92, 0.94],
#     'val_accuracy': [0.82, 0.87, 0.89, 0.91, 0.9481],
#     'loss': [0.45, 0.38, 0.33, 0.28, 0.22],
#     'val_loss': [0.48, 0.40, 0.36, 0.30, 0.25]
# }

# # -------------------------
# # Accuracy Plot
# # -------------------------
# plt.figure(figsize=(10, 5))
# plt.plot(history['accuracy'], marker='o', label='Training Accuracy', color='navy')
# plt.plot(history['val_accuracy'], marker='o', label='Validation Accuracy', color='crimson')
# plt.title('Model Accuracy', fontsize=16, fontweight='bold')
# plt.xlabel('Epoch', fontsize=12, fontweight='bold')
# plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig('accuracy_plot.png')  # Save the figure
# plt.show()

# # -------------------------
# # Loss Plot
# # -------------------------
# plt.figure(figsize=(10, 5))
# plt.plot(history['loss'], marker='o', label='Training Loss', color='navy')
# plt.plot(history['val_loss'], marker='o', label='Validation Loss', color='crimson')
# plt.title('Model Loss', fontsize=16, fontweight='bold')
# plt.xlabel('Epoch', fontsize=12, fontweight='bold')
# plt.ylabel('Loss', fontsize=12, fontweight='bold')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig('loss_plot.png')  # Save the figure
# plt.show()
