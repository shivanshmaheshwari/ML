import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.utils import FeatureSpace

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
        "cp": FeatureSpace.integer_categorical(num_oov_indices=0),
        "fbs": FeatureSpace.integer_categorical(num_oov_indices=0),
        "restecg": FeatureSpace.integer_categorical(num_oov_indices=0),
        "exang": FeatureSpace.integer_categorical(num_oov_indices=0),
        "ca": FeatureSpace.integer_categorical(num_oov_indices=0),
        # Categorical feature encoded as string
        "thal": FeatureSpace.string_categorical(num_oov_indices=0),
        # Numerical features to normalize
        "age": FeatureSpace.float_discretized(num_bins=30),
        # Numerical features to normalize
        "trestbps": FeatureSpace.float_normalized(),
        "chol": FeatureSpace.float_normalized(),
        "thalach": FeatureSpace.float_normalized(),
        "oldpeak": FeatureSpace.float_normalized(),
        "slope": FeatureSpace.float_normalized(),
    },
    # Specify feature cross with a custom crossing dim.
    crosses=[
        FeatureSpace.cross(feature_names=("sex", "age"), crossing_dim=64),
        FeatureSpace.cross(
            feature_names=("thal", "ca"),
            crossing_dim=16,
        ),
    ],
    output_mode="concat",
)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(32, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
training_model.fit(
    preprocessed_train_ds, epochs=20, validation_data=preprocessed_val_ds, verbose=2
)

sample = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = inference_model.predict(input_dict)

print(
    f"This particular patient had a {100 * predictions[0][0]:.2f}% probability "
    "of having a heart disease, as evaluated by our model."
)