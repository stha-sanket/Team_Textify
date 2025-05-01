import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Import necessary components for Transfer Learning and Callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns # Added import for heatmap
from sklearn.metrics import confusion_matrix, classification_report # Added sklearn imports
import pickle # Import is here, but model.save is used for saving the model
import numpy as np
import os
import zipfile
import pathlib
import shutil

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Configuration ---
# Use raw string literal (r'') or double backslashes for Windows paths
zip_file_path = r'/home/sanket/Desktop/hackaton/final/ImagePDFClassification.zip' # <--- VERIFY THIS PATH
extract_dir = 'temp_image_data_tl_ft'    # Use a different temp dir name
img_height = 224                         # MobileNetV2 default input size
img_width = 224
batch_size = 32
initial_epochs = 75                      # Epochs for initial transfer learning phase
fine_tune_epochs = 50                    # Epochs for the fine-tuning phase
validation_split = 0.2
initial_learning_rate = 0.0001           # LR for initial phase
fine_tune_learning_rate = initial_learning_rate / 10 # LR for fine-tuning (e.g., 1e-5)
fine_tune_at_layer = 100                 # Unfreeze layers from this index onwards (adjust as needed)
                                         # Use 0 to unfreeze all base model layers

# --- 2. Extract the Zip File ---
if os.path.exists(extract_dir):
    print(f"Removing existing directory: {extract_dir}")
    shutil.rmtree(extract_dir)

print(f"Extracting {zip_file_path} to {extract_dir}...")
try:
    # Ensure the directory for extraction exists
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

    # --- CORRECTED Logic to find the data directory ---
    extracted_items = os.listdir(extract_dir)
    if not extracted_items:
        raise ValueError("Extraction resulted in an empty directory.")

    potential_data_dir_item_path = os.path.join(extract_dir, extracted_items[0])

    # If there's exactly one item and it's a directory, assume it's the container
    if len(extracted_items) == 1 and os.path.isdir(potential_data_dir_item_path):
        data_dir = pathlib.Path(potential_data_dir_item_path)
        print(f"Adjusted data directory (using single top-level folder): {data_dir}")
    else:
        # Otherwise, assume class folders are directly inside extract_dir
        data_dir = pathlib.Path(extract_dir)
        print(f"Using data directory (expected class folders directly inside): {data_dir}")
    # --- END CORRECTED Logic ---

except FileNotFoundError:
    print(f"Error: Zip file not found at {zip_file_path}")
    exit()
except zipfile.BadZipFile:
    print(f"Error: Bad Zip file - {zip_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred during extraction or path adjustment: {e}")
    exit()

# Count images robustly
image_count = sum(len(list(data_dir.glob(f'*/*.{ext}'))) for ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif'])
print(f"Found {image_count} images in subdirectories of {data_dir}")
if image_count == 0:
    print(f"Error: No image files found in the class subdirectories of {data_dir}")
    print("Contents of data_dir:")
    try:
        for item in data_dir.glob('*'):
            print(f"- {item} {'(DIR)' if item.is_dir() else '(FILE)'}")
    except Exception as list_e:
        print(f"Could not list contents of {data_dir}: {list_e}")
    exit()
# --- END OF SECTION 2 ---

# --- 3. Load Data using Keras utility ---
print("Loading training data...")
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size,
      label_mode='categorical'
    )

    print("Loading validation data...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="validation",
      seed=123, # Use the same seed!
      image_size=(img_height, img_width),
      batch_size=batch_size,
      label_mode='categorical'
    )
except Exception as e:
    print(f"Error loading data with image_dataset_from_directory: {e}")
    print(f"Please check the directory structure within {data_dir}. It should contain subdirectories named after your classes, with images inside them.")
    exit()


class_names = train_ds.class_names
num_classes = len(class_names)
if num_classes <= 1:
    print(f"Error: Found {num_classes} classes ({class_names}). Need at least 2 classes for classification.")
    exit()
print(f"Found classes: {class_names}")
print(f"Number of classes: {num_classes}")

# --- 4. Configure Dataset for Performance ---
AUTOTUNE = tf.data.AUTOTUNE

# --- 5. Data Augmentation Layer ---
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
  ],
  name="data_augmentation"
)

# Apply augmentation and preprocessing
def prepare(ds, shuffle=False, augment=False):
    # Apply caching first
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000)

    # Apply augmentation if needed
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Apply preprocessing
    ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
                num_parallel_calls=AUTOTUNE)

    # Apply prefetching
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds) # Prepare validation set (no shuffle/augment)

# --- 6. Build the Transfer Learning Model (MobileNetV2) ---
print("Building model with MobileNetV2 base...")

inputs = Input(shape=(img_height, img_width, 3), name="input_layer")
base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')

# --- Initial Phase: Freeze the base model ---
base_model.trainable = False
print("Base model frozen for initial training.")
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D(name="global_avg_pool")(x)
x = Dropout(0.3, name="top_dropout")(x)
outputs = Dense(num_classes, activation='softmax', name="output_layer")(x)
model = Model(inputs, outputs)

# --- 7. Compile the Model for Initial Training ---
print("Compiling model for initial transfer learning phase...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

# --- 8. Define Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)

# --- 9. Initial Training Phase ---
print(f"\n--- Starting Initial Training Phase (up to {initial_epochs} epochs) ---")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=initial_epochs,
  callbacks=[early_stopping, reduce_lr]
)
print("Initial training phase finished.")
initial_epochs_run = len(history.history['loss'])

# --- 10. Evaluate After Initial Training ---
print("\nEvaluating model after initial training (best weights)...")
loss0, accuracy0 = model.evaluate(val_ds)
print(f"Initial Validation Loss: {loss0:.4f}")
print(f"Initial Validation Accuracy: {accuracy0:.4f} ({accuracy0*100:.2f}%)")

# --- 11. Fine-Tuning Phase ---
print(f"\n--- Starting Fine-Tuning Phase (up to {fine_tune_epochs} epochs) ---")
base_model.trainable = True
if fine_tune_at_layer > 0:
    print(f"Unfreezing layers from index {fine_tune_at_layer} of the base model onwards.")
    for layer in base_model.layers[:fine_tune_at_layer]:
        layer.trainable = False
else:
     print("Unfreezing all layers of the base model.")

print("Re-compiling model for fine-tuning with lower learning rate...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

total_epochs_potential = initial_epochs + fine_tune_epochs
history_fine = model.fit(
    train_ds,
    epochs=total_epochs_potential,
    initial_epoch=initial_epochs_run,
    validation_data=val_ds,
    callbacks=[early_stopping, reduce_lr] # Reuse callbacks
)
print("Fine-tuning phase finished.")

# --- 12. Evaluate After Fine-Tuning ---
print("\nEvaluating model after fine-tuning (best weights from entire process)...")
loss_final, accuracy_final = model.evaluate(val_ds)
print(f"Final Validation Loss: {loss_final:.4f}")
print(f"Final Validation Accuracy: {accuracy_final:.4f} ({accuracy_final*100:.2f}%)")


# --- START: ADDED SECTION for Confusion Matrix, Report, Saving ---

print("\n--- Generating Confusion Matrix and Classification Report ---")

# 1. Get Predictions and True Labels for the Validation Set
y_pred = []  # Store predicted labels
y_true = []  # Store true labels

# Iterate over the validation dataset
# Ensure val_ds is available and has data
if val_ds is None:
    print("Error: Validation dataset (val_ds) is not available for generating report.")
else:
    # Calculate dataset size more accurately if possible
    val_cardinality = tf.data.experimental.cardinality(val_ds)
    if val_cardinality == tf.data.experimental.INFINITE_CARDINALITY:
        print("Warning: Validation dataset size is infinite (maybe repeated?). Calculating size...")
        dataset_size = 0
        for _ in val_ds:
            dataset_size += 1
    elif val_cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
        print("Warning: Validation dataset size is unknown. Calculating size...")
        dataset_size = 0
        for _ in val_ds:
            dataset_size += 1
    else:
        dataset_size = val_cardinality.numpy()

    if dataset_size == 0:
         print("Error: Validation dataset (val_ds) is empty.")
    else:
        num_val_samples = dataset_size * batch_size # Approx, might be less in last batch
        print(f"Generating predictions for approx {num_val_samples} validation samples...")
        for images, labels in val_ds:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            preds = model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))

        # Convert lists to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Check if we got predictions (adjust expected count based on actual retrieval)
        retrieved_count = len(y_true)
        if retrieved_count == 0:
             print("Error: Could not retrieve labels/predictions from validation set.")
        else:
            print(f"Retrieved {retrieved_count} predictions for evaluation.")

            # 2. Calculate Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)

            # 3. Plot Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show() # Display the plot

            # 4. Calculate and Print Classification Report
            print("\nClassification Report:")
            try:
                # Ensure labels cover the range defined by class_names if necessary
                report_labels = list(range(num_classes))
                report = classification_report(y_true, y_pred, labels=report_labels, target_names=class_names, zero_division=0)
                print(report)
            except ValueError as e:
                print(f"Could not generate classification report: {e}")
                print("This might happen if some classes expected based on folders were not present in the validation split.")

            # Also print the overall accuracy again (should match model.evaluate result)
            overall_accuracy = np.sum(y_true == y_pred) / len(y_true)
            print(f"\nOverall Accuracy (calculated from predictions): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

# --- 5. Save the Final Model ---
# <<< CORRECTED model save path with .keras extension >>>
model_save_path = 'my_image_classifier_model.keras' # Keras native format (single file)

print(f"\n--- Saving the trained model to: {model_save_path} ---")
try:
    model.save(model_save_path)
    # <<< UPDATED success print message >>>
    print(f"Model successfully saved in Keras native format at '{model_save_path}'")

except Exception as e:
    print(f"Error saving model: {e}")

# --- END: ADDED SECTION ---


# --- 13. Visualize Combined Training Results --- # Renumbered section
# Combine history data
# Ensure history_fine exists, otherwise use only history
if 'history_fine' in locals():
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    fine_tune_epochs_run = len(history_fine.history['loss'])
else:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fine_tune_epochs_run = 0 # No fine tuning epochs run

# Total actual epochs run
total_epochs_run = len(acc)
epochs_range = range(total_epochs_run)

plt.figure(figsize=(14, 6)) # Adjusted figure size slightly

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
if 'history_fine' in locals(): # Only plot line if fine-tuning happened
    plt.axvline(initial_epochs_run - 1, linestyle='--', color='gray', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Combined Training and Validation Accuracy')
plt.xlabel(f'Epoch (Initial: {initial_epochs_run}, Fine-Tune: {fine_tune_epochs_run})')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6) # Added grid

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
if 'history_fine' in locals(): # Only plot line if fine-tuning happened
    plt.axvline(initial_epochs_run - 1, linestyle='--', color='gray', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Combined Training and Validation Loss')
plt.xlabel(f'Epoch (Initial: {initial_epochs_run}, Fine-Tune: {fine_tune_epochs_run})')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6) # Added grid

plt.tight_layout()
plt.show() # Display the plot

print("\nScript finished (including evaluation report and model saving).")