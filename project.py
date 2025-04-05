import os
import numpy as np
import cv2
import albumentations as A
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import keras_tuner as kt

# Data Loading 

def load_dataset(dataset_dir):
    def gather_images_labels(directory, label):
        image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory)]
        labels = [label] * len(image_paths)
        return image_paths, labels

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    valid_dir = os.path.join(dataset_dir, 'valid')

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    valid_paths, valid_labels = [], []

    for fname in os.listdir(train_dir):
        label = 0 if "Non_Autistic" in fname else 1
        train_paths.append(os.path.join(train_dir, fname))
        train_labels.append(label)

    for fname in os.listdir(test_dir):
        label = 0 if "Non_Autistic" in fname else 1
        test_paths.append(os.path.join(test_dir, fname))
        test_labels.append(label)

    for fname in os.listdir(os.path.join(valid_dir, 'Autistic')):
        valid_paths.append(os.path.join(valid_dir, 'Autistic', fname))
        valid_labels.append(1)
    for fname in os.listdir(os.path.join(valid_dir, 'Non_Autistic')):
        valid_paths.append(os.path.join(valid_dir, 'Non_Autistic', fname))
        valid_labels.append(0)

    return (np.array(train_paths), np.array(train_labels)), \
           (np.array(test_paths), np.array(test_labels)), \
           (np.array(valid_paths), np.array(valid_labels))

# Data augmentation and preprocessing
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=40, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
])

def preprocess_image(image_path, augment=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    if augment:
        img = transform(image=img)['image']
    return img.astype('float32') / 255.0

def process_images(image_paths, augment=False):
    return np.array([preprocess_image(p, augment) for p in image_paths])

# Model Definition
# Freeze the base model layers and add custom layers

def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(hp.Int('dense_units', 128, 512, step=64), activation='relu')(x)
    x = layers.Dropout(hp.Float('dropout_rate', 0.3, 0.6, step=0.1))(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load Data

dataset_dir = '/home/reyhaneh.farahmand/software/src/enel645_project/AutismDataset/AutismDataset'
(train_paths, train_labels), (test_paths, test_labels), (valid_paths, valid_labels) = load_dataset(dataset_dir)

train_paths, train_labels = shuffle(train_paths, train_labels, random_state=42)
test_paths, test_labels = shuffle(test_paths, test_labels, random_state=42)
valid_paths, valid_labels = shuffle(valid_paths, valid_labels, random_state=42)

# Hyperparameter Tuning with KerasTuner 

X_train_tune = process_images(train_paths[:500], augment=True)
y_train_tune = train_labels[:500] 

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='autism_tuner',
    project_name='vgg_tune'
)

tuner.search(X_train_tune, y_train_tune, epochs=4, validation_split=0.2, verbose=1)
best_hp = tuner.get_best_hyperparameters(1)[0]

print("\nBest Hyperparameters:")
print(f"Learning Rate: {best_hp.get('learning_rate')}")
print(f"Dense Units: {best_hp.get('dense_units')}")
print(f"Dropout Rate: {best_hp.get('dropout_rate')}")

#  K-Fold Cross-Validation 

kf = KFold(n_splits=3, shuffle=True, random_state=42)
X_train_all = process_images(train_paths)
y_train_all = train_labels

accuracy_list, precision_list, recall_list, f1_list, roc_auc_list = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_all)):
    print(f"\nTraining Fold {fold + 1}...")

    X_fold_train = X_train_all[train_idx]
    y_fold_train = y_train_all[train_idx]
    X_fold_val = X_train_all[val_idx]
    y_fold_val = y_train_all[val_idx]

    model = build_model(best_hp)

    checkpoint = ModelCheckpoint(f'best_model_fold{fold+1}.h5', save_best_only=True)
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(
        X_fold_train, y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=10,
        batch_size=16,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    y_pred = model.predict(X_fold_val).flatten()
    y_pred_label = (y_pred > 0.5).astype(int)

    accuracy_list.append(accuracy_score(y_fold_val, y_pred_label))
    precision_list.append(precision_score(y_fold_val, y_pred_label))
    recall_list.append(recall_score(y_fold_val, y_pred_label))
    f1_list.append(f1_score(y_fold_val, y_pred_label))
    roc_auc_list.append(roc_auc_score(y_fold_val, y_pred))


print("\nK-Fold Cross-Validation Results:")
print(f"Accuracy: {np.mean(accuracy_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f}")
print(f"Recall: {np.mean(recall_list):.4f}")
print(f"F1 Score: {np.mean(f1_list):.4f}")
print(f"AUC-ROC: {np.mean(roc_auc_list):.4f}")

# Final Model Evaluation 

final_model = build_model(best_hp)
final_model.fit(
    X_train_all, y_train_all,
    epochs=5,
    batch_size=8,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)

X_test = process_images(test_paths)
y_pred_test = final_model.predict(X_test).flatten()
y_pred_label_test = (y_pred_test > 0.5).astype(int)

print("\nTest Set Evaluation:")
print("Accuracy:", accuracy_score(test_labels, y_pred_label_test))
print("Precision:", precision_score(test_labels, y_pred_label_test))
print("Recall:", recall_score(test_labels, y_pred_label_test))
print("F1 Score:", f1_score(test_labels, y_pred_label_test))
print("AUC-ROC:", roc_auc_score(test_labels, y_pred_test))

final_model.save("autism_final_model.h5")
