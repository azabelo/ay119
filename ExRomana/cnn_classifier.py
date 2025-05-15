import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# additional imports for metrics and plotting
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from astropy.visualization import ZScaleInterval

# Load training samples data
base_dir = "."
input_file = os.path.join(base_dir, "exromana_training_samples.npy")
data = np.load(input_file, allow_pickle=True).item()

tp_samples = data.get("TP", [])
fp_samples = data.get("FP", [])

# Extract the fourth layer (scorr channel) from sample["sample"] (64x64x4 array)
samples = []
labels = []
for sample in tp_samples:
    samples.append(sample["sample"][..., 3])
    labels.append(1)

for sample in fp_samples:
    samples.append(sample["sample"][..., 3])
    labels.append(0)

X = np.array(samples)
y = np.array(labels)

# Ensure images have a channel dimension (now grayscale images of the scorr layer)
if len(X.shape) == 3:  # shape (N, height, width)
    X = np.expand_dims(X, -1)

# Normalize the data using ZScale: for each sample compute limits and scale to [0,255]
zscale = ZScaleInterval()
X_z = np.empty_like(X, dtype=np.float32)
for i in range(X.shape[0]):
    # Remove channel dimension for zscale calculation
    img = X[i, ..., 0]
    vmin, vmax = zscale.get_limits(img)
    # Linear scaling to [0,255]
    scaled = (img - vmin) * (255.0 / (vmax - vmin))
    X_z[i, ..., 0] = np.clip(scaled, 0, 255)
X = X_z.astype(np.uint8)

print("After ZScale normalization - min:", np.min(X), "max:", np.max(X))

# Split data: first extract test set (15%), then split remaining into training and validation.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
val_ratio = 0.15 / 0.85  # relative validation size from remaining data
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)

# Build small CNN using only the scorr layer
input_shape = X_train.shape[1:]
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Added an extra conv layer with fewer filters followed by pooling
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

num_epochs = 20
# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(X_val, y_val))

# Display training and validation losses.
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(base_dir, "validation_loss_{num_epochs}.png"))
plt.tight_layout()
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Compute predictions, confusion matrix, and classification report.
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix.
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["FP", "TP"], rotation=45)
plt.yticks(tick_marks, ["FP", "TP"])

# Annotate each cell with its value.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2. else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(base_dir, "validation_CM_{num_epochs}.png"))
plt.tight_layout()
plt.show()
