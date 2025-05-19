import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

train_path = 'data/training_set'
test_path = 'data/test_set'

image_size = (128, 128)
batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=False
)

class_names = train_dataset.class_names
print("Class names:", class_names)

val_size = int(0.2 * tf.data.experimental.cardinality(train_dataset).numpy())
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))


def prepare_for_sklearn(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        images.extend(image_batch.numpy().reshape(image_batch.shape[0], -1))
        labels.extend(np.squeeze(label_batch.numpy()))
    return np.array(images), np.array(labels)


train_images_flat, train_labels = prepare_for_sklearn(train_dataset)
val_images_flat, val_labels = prepare_for_sklearn(val_dataset)
test_images_flat, test_labels = prepare_for_sklearn(test_dataset)

scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images_flat)
val_images_scaled = scaler.transform(val_images_flat)
test_images_scaled = scaler.transform(test_images_flat)

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(
    train_dataset, validation_data=val_dataset, epochs=20, verbose=1)
cnn_loss, cnn_accuracy = cnn_model.evaluate(test_dataset, verbose=0)
cnn_preds_probs = cnn_model.predict(test_dataset, verbose=0)
cnn_preds = (cnn_preds_probs > 0.5).astype(int).flatten()

print("Model CNN:")
print(f"Test Accuracy: {cnn_accuracy:.4f}")
print(f"Test Loss: {cnn_loss:.4f}")
print(classification_report(test_labels, cnn_preds, target_names=class_names))
cm_cnn = confusion_matrix(test_labels, cnn_preds)

cnn_complex_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn_complex_model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_complex_history = cnn_complex_model.fit(
    train_dataset, validation_data=val_dataset, epochs=20, verbose=1)
cnn_complex_loss, cnn_complex_accuracy = cnn_complex_model.evaluate(
    test_dataset, verbose=0)
cnn_complex_preds_probs = cnn_complex_model.predict(test_dataset, verbose=0)
cnn_complex_preds = (cnn_complex_preds_probs > 0.5).astype(int).flatten()

print("\nModel CNN (bardziej złożony):")
print(f"Test Accuracy: {cnn_complex_accuracy:.4f}")
print(f"Test Loss: {cnn_complex_loss:.4f}")
print(classification_report(test_labels,
                            cnn_complex_preds, target_names=class_names))
cm_cnn_complex = confusion_matrix(test_labels, cnn_complex_preds)

base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

transfer_learning_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

transfer_learning_model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
transfer_learning_history = transfer_learning_model.fit(
    train_dataset, validation_data=val_dataset, epochs=20, verbose=1)
transfer_learning_loss, transfer_learning_accuracy = transfer_learning_model.evaluate(
    test_dataset, verbose=0)
transfer_learning_preds_probs = transfer_learning_model.predict(
    test_dataset, verbose=0)
transfer_learning_preds = (
    transfer_learning_preds_probs > 0.5).astype(int).flatten()

print("\nModel CNN (Transfer Learning - MobileNetV2):")
print(f"Test Accuracy: {transfer_learning_accuracy:.4f}")
print(f"Test Loss: {transfer_learning_loss:.4f}")
print(classification_report(test_labels,
                            transfer_learning_preds, target_names=class_names))
cm_transfer_learning = confusion_matrix(test_labels, transfer_learning_preds)

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def plot_confusion_matrix(cm, class_names, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


plot_confusion_matrix(cm_cnn, class_names,
                      'Confusion Matrix (CNN)', 'cm_cnn.png')
plot_confusion_matrix(cm_cnn_complex, class_names,
                      'Confusion Matrix (CNN - Złożony)', 'cm_cnn_complex.png')
plot_confusion_matrix(cm_transfer_learning, class_names,
                      'Confusion Matrix (Transfer Learning - MobileNetV2)', 'cm_transfer_learning.png')


def plot_learning_curves(history, title, filename):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


plot_learning_curves(cnn_history, 'Learning Curves (CNN)',
                     'learning_curves_cnn.png')
plot_learning_curves(cnn_complex_history,
                     'Learning Curves (CNN - Złożony)', 'learning_curves_cnn_complex.png')
plot_learning_curves(transfer_learning_history,
                     'Learning Curves (Transfer Learning - MobileNetV2)', 'learning_curves_transfer_learning.png')

plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):
    preds = transfer_learning_model.predict(images)
    pred_labels = (preds > 0.5).astype(int).flatten()
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(
            f"True: {class_names[int(labels[i])]}, Pred: {class_names[pred_labels[i]]}")
        plt.axis("off")
plt.tight_layout()
plt.suptitle("Przykładowe Predykcje (Transfer Learning - MobileNetV2)", y=1.02)
plt.savefig(os.path.join(
    output_dir, 'sample_predictions_transfer_learning.png'))
plt.close()

print("\n--- Wybór Najlepszego Modelu ---")
print("Porównanie dokładności testowej:")
print(f"CNN: {cnn_accuracy:.4f}")
print(f"CNN (Złożony): {cnn_complex_accuracy:.4f}")
print(f"Transfer Learning (MobileNetV2): {transfer_learning_accuracy:.4f}")

best_model_name = ""
best_accuracy = 0.0

if cnn_accuracy > best_accuracy:
    best_accuracy = cnn_accuracy
    best_model_name = "CNN"
if cnn_complex_accuracy > best_accuracy:
    best_accuracy = cnn_complex_accuracy
    best_model_name = "CNN (Złożony)"
if transfer_learning_accuracy > best_accuracy:
    best_accuracy = transfer_learning_accuracy
    best_model_name = "Transfer Learning (MobileNetV2)"

print(
    f"\nNajlepszym modelem na podstawie dokładności testowej jest: {best_model_name} z dokładnością: {best_accuracy:.4f}")
