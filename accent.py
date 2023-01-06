import os
import io
import csv
import random

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from IPython.display import Audio

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt

SEED = 1337
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_RATIO = 0.1
MODEL_NAME = "uk_irish_accent_recognition"
CACHE_DIR = None
URL_PATH = "https://www.openslr.org/resources/83/"

# List of datasets compressed files that contain the audio files
zip_files = {
    0: "irish_english_male.zip",
    1: "midlands_english_female.zip",
    2: "midlands_english_male.zip",
    3: "northern_english_female.zip",
    4: "northern_english_male.zip",
    5: "scottish_english_female.zip",
    6: "scottish_english_male.zip",
    7: "southern_english_female.zip",
    8: "southern_english_male.zip",
    9: "welsh_english_female.zip",
    10: "welsh_english_male.zip",
}
# List of gender agnostic categories
gender_agnostic_categories = [
    "ir",  # Irish
    "mi",  # Midlands
    "no",  # Northern
    "sc",  # Scottish
    "so",  # Southern
    "we",  # Welsh
]
# List of class names
class_names = [
    "Irish",
    "Midlands",
    "Northern",
    "Scottish",
    "Southern",
    "Welsh",
    "Not a speech",
]

# Set all random seeds in order to get reproducible results
keras.utils.set_random_seed(SEED)
# Where to download the dataset
DATASET_DESTINATION = os.path.join(CACHE_DIR if CACHE_DIR else "~/.keras/", "datasets")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
line_index_file = keras.utils.get_file(
    fname="line_index_file", origin=URL_PATH + "line_index_all.csv"
)
# Download the list of compressed files that contains the audio wav files
for i in zip_files:
    fname = zip_files[i].split(".")[0]
    url = URL_PATH + zip_files[i]

    zip_file = keras.utils.get_file(fname=fname, origin=url, extract=True)
    os.remove(zip_file)

# Load the data in a Dataframe
dataframe = pd.read_csv(
    line_index_file, names=["id", "filename", "transcript"], usecols=["filename"]
)
dataframe.head()

def preprocess_dataframe(dataframe):
    # Remove leading space in filename column
    dataframe["filename"] = dataframe.apply(lambda row: row["filename"].strip(), axis=1)
    # Create gender agnostic labels based on the filename first 2 letters
    dataframe["label"] = dataframe.apply(
        lambda row: gender_agnostic_categories.index(row["filename"][:2]), axis=1
    )
    # Add the file path to the name
    dataframe["filename"] = dataframe.apply(
        lambda row: os.path.join(DATASET_DESTINATION, row["filename"] + ".wav"), axis=1
    )
    # Shuffle the samples
    dataframe = dataframe.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return dataframe

dataframe = preprocess_dataframe(dataframe)
dataframe.head()

# Prepare training & validation sets
split = int(len(dataframe) * (1 - VALIDATION_RATIO))
train_df = dataframe[:split]
valid_df = dataframe[split:]
print(
    f"We have {train_df.shape[0]} training samples & {valid_df.shape[0]} validation ones"
)

# Prepare a TensorFlow Dataset
@tf.function
def load_16k_audio_wav(filename):
    # Read file content
    file_content = tf.io.read_file(filename)
    # Decode audio wave
    audio_wav, sample_rate = tf.audio.decode_wav(file_content, desired_channels=1)
    audio_wav = tf.squeeze(audio_wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16k
    audio_wav = tfio.audio.resample(audio_wav, rate_in=sample_rate, rate_out=16000)

    return audio_wav

def filepath_to_embeddings(filename, label):
    # Load 16k audio wave
    audio_wav = load_16k_audio_wav(filename)
    # Get audio embeddings & scores.
    # The embeddings are the audio features extracted using transfer learning
    # while scores will be used to identify time slots that are not speech
    # which will then be gathered into a specific new category 'other'
    scores, embeddings, _ = yamnet_model(audio_wav)
    # Number of embeddings in order to know how many times to repeat the label
    embeddings_num = tf.shape(embeddings)[0]
    labels = tf.repeat(label, embeddings_num)
    # Change labels for time-slots that are not speech into a new category 'other'
    labels = tf.where(tf.argmax(scores, axis=1) == 0, label, len(class_names) - 1)
    # Using one-hot in order to use AUC
    return (embeddings, tf.one_hot(labels, len(class_names)))

def dataframe_to_dataset(dataframe, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["filename"], dataframe["label"])
    )
    dataset = dataset.map(
        lambda x, y: filepath_to_embeddings(x, y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).unbatch()

    return dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = dataframe_to_dataset(train_df)
valid_ds = dataframe_to_dataset(valid_df)

# Build the model
keras.backend.clear_session()
def build_and_compile_model():
    inputs = keras.layers.Input(shape=(1024), name="embedding")

    x = keras.layers.Dense(256, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dropout(0.15, name="dropout_1")(x)
    x = keras.layers.Dense(384, activation="relu", name="dense_2")(x)
    x = keras.layers.Dropout(0.2, name="dropout_2")(x)
    x = keras.layers.Dense(192, activation="relu", name="dense_3")(x)
    x = keras.layers.Dropout(0.25, name="dropout_3")(x)
    x = keras.layers.Dense(384, activation="relu", name="dense_4")(x)
    x = keras.layers.Dropout(0.2, name="dropout_4")(x)

    outputs = keras.layers.Dense(len(class_names), activation="softmax", name="ouput")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="accent_recognition")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1.9644e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model

model = build_and_compile_model()
model.summary()

# Class weights calculation
class_counts = tf.zeros(shape=(len(class_names),), dtype=tf.int32)
for x, y in iter(train_ds):
    class_counts = class_counts + tf.math.bincount(
        tf.cast(tf.math.argmax(y, axis=1), tf.int32), minlength=len(class_names)
    )
class_weight = {
    i: tf.math.reduce_sum(class_counts).numpy() / class_counts[i].numpy()
    for i in range(len(class_counts))
}
print(class_weight)

# Callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=10, restore_best_weights=True
)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
    MODEL_NAME + ".h5", monitor="val_auc", save_best_only=True
)
tensorboard_cb = keras.callbacks.TensorBoard(
    os.path.join(os.curdir, "logs", model.name)
)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

# Training
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2,
)

# Results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
axs[0].plot(range(EPOCHS), history.history["accuracy"], label="Training")
axs[0].plot(range(EPOCHS), history.history["val_accuracy"], label="Validation")
axs[0].set_xlabel("Epochs")
axs[0].set_title("Training & Validation Accuracy")
axs[0].legend()
axs[0].grid(True)
axs[1].plot(range(EPOCHS), history.history["auc"], label="Training")
axs[1].plot(range(EPOCHS), history.history["val_auc"], label="Validation")
axs[1].set_xlabel("Epochs")
axs[1].set_title("Training & Validation AUC")
axs[1].legend()
axs[1].grid(True)
plt.show()

# Evaluation
train_loss, train_acc, train_auc = model.evaluate(train_ds)
valid_loss, valid_acc, valid_auc = model.evaluate(valid_ds)

# The following function calculates the d-prime score from the AUC
def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

print(
    "train d-prime: {0:.3f}, validation d-prime: {1:.3f}".format(
        d_prime(train_auc), d_prime(valid_auc)
    )
)

# Create x and y tensors
x_valid = None
y_valid = None

for x, y in iter(valid_ds):
    if x_valid is None:
        x_valid = x.numpy()
        y_valid = y.numpy()
    else:
        x_valid = np.concatenate((x_valid, x.numpy()), axis=0)
        y_valid = np.concatenate((y_valid, y.numpy()), axis=0)

# Generate predictions
y_pred = model.predict(x_valid)
# Calculate confusion matrix
confusion_mtx = tf.math.confusion_matrix(
    np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1)
)
# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx, xticklabels=class_names, yticklabels=class_names, annot=True, fmt="g"
)
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.title("Validation Confusion Matrix")
plt.show()

# Precision & recall
for i, label in enumerate(class_names):
    precision = confusion_mtx[i, i] / np.sum(confusion_mtx[:, i])
    recall = confusion_mtx[i, i] / np.sum(confusion_mtx[i, :])
    print(
        "{0:15} Precision:{1:.2f}%; Recall:{2:.2f}%".format(
            label, precision * 100, recall * 100
        )
    )

# Run inference on test data
filename = "audio-sample-Stuart"
url = "https://www.thescottishvoice.org.uk/files/cm/files/"

if os.path.exists(filename + ".wav") == False:
    print(f"Downloading {filename}.mp3 from {url}")
    command = f"wget {url}{filename}.mp3"
    os.system(command)

    print(f"Converting mp3 to wav and resampling to 16 kHZ")
    command = (
        f"ffmpeg -hide_banner -loglevel panic -y -i {filename}.mp3 -acodec "
        f"pcm_s16le -ac 1 -ar 16000 {filename}.wav"
    )
    os.system(command)

filename = filename + ".wav"

def yamnet_class_names_from_csv(yamnet_class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    yamnet_class_map_csv = io.StringIO(yamnet_class_map_csv_text)
    yamnet_class_names = [
        name for (class_index, mid, name) in csv.reader(yamnet_class_map_csv)
    ]
    yamnet_class_names = yamnet_class_names[1:]  # Skip CSV header
    return yamnet_class_names

yamnet_class_map_path = yamnet_model.class_map_path().numpy()
yamnet_class_names = yamnet_class_names_from_csv(
    tf.io.read_file(yamnet_class_map_path).numpy().decode("utf-8")
)

def calculate_number_of_non_speech(scores):
    number_of_non_speech = tf.math.reduce_sum(
        tf.where(tf.math.argmax(scores, axis=1, output_type=tf.int32) != 0, 1, 0)
    )
    return number_of_non_speech

def filename_to_predictions(filename):
    # Load 16k audio wave
    audio_wav = load_16k_audio_wav(filename)
    # Get audio embeddings & scores.
    scores, embeddings, mel_spectrogram = yamnet_model(audio_wav)
    print(
        "Out of {} samples, {} are not speech".format(
            scores.shape[0], calculate_number_of_non_speech(scores)
        )
    )
    # Predict the output of the accent recognition model with embeddings as input
    predictions = model.predict(embeddings)

    return audio_wav, predictions, mel_spectrogram

# Let's run the model on the audio file:
audio_wav, predictions, mel_spectrogram = filename_to_predictions(filename)
infered_class = class_names[predictions.mean(axis=0).argmax()]
print(f"The main accent is: {infered_class} English")
# Listen to the audio
Audio(audio_wav, rate=16000)
plt.figure(figsize=(10, 6))
# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(audio_wav)
plt.xlim([0, len(audio_wav)])
# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.imshow(
    mel_spectrogram.numpy().T, aspect="auto", interpolation="nearest", origin="lower"
)
# Plot and label the model output scores for the top-scoring classes.
mean_predictions = np.mean(predictions, axis=0)
top_class_indices = np.argsort(mean_predictions)[::-1]
plt.subplot(3, 1, 3)
plt.imshow(
    predictions[:, top_class_indices].T,
    aspect="auto",
    interpolation="nearest",
    cmap="gray_r",
)
# patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
# values from the model documentation
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding - 0.5, predictions.shape[0] + patch_padding - 0.5])
# Label the top_N classes.
yticks = range(0, len(class_names), 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
_ = plt.ylim(-0.5 + np.array([len(class_names), 0]))