import os
import nltk
import logging
import keras_nlp

import numpy as np
import tensorflow as tf
from tensorflow import keras
from datasets import load_dataset

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers.keras_callbacks import KerasMetricCallback

# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRAIN_TEST_SPLIT = 0.1 # 10% of the dataset will be used for training and testing
MAX_INPUT_LENGTH = 1024  # Maximum length of the input to the model
MIN_TARGET_LENGTH = 5  # Minimum length of the output by the model
MAX_TARGET_LENGTH = 128  # Maximum length of the output by the model
BATCH_SIZE = 8  # Batch-size for training our model
LEARNING_RATE = 2e-5  # Learning-rate for training our model
MAX_EPOCHS = 1  # Maximum number of epochs we will train the model for
MODEL_CHECKPOINT = "t5-small"

# Load the dataset
raw_datasets = load_dataset("xsum", split="train")
raw_datasets = raw_datasets.train_test_split(
    train_size=TRAIN_TEST_SPLIT, test_size=TRAIN_TEST_SPLIT
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
if MODEL_CHECKPOINT in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

train_dataset = tokenized_datasets["train"].to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    collate_fn=data_collator,
)
test_dataset = tokenized_datasets["test"].to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=False,
    collate_fn=data_collator,
)
generation_dataset = (
    tokenized_datasets["test"]
    .shuffle()
    .select(list(range(200)))
    .to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer)
rouge_l = keras_nlp.metrics.RougeL()

def metric_fn(eval_predictions):
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_l(decoded_labels, decoded_predictions)
    # We will print only the F1 score, you can use other aggregation metrics as well
    result = {"RougeL": result["f1_score"]}

    return result

metric_callback = KerasMetricCallback(
    metric_fn, eval_dataset=generation_dataset, predict_with_generate=True
)
callbacks = [metric_callback]
model.fit(
    train_dataset, validation_data=test_dataset, epochs=MAX_EPOCHS, callbacks=callbacks
)

# Inference
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")
summarizer(
    raw_datasets["test"][0]["document"],
    min_length=MIN_TARGET_LENGTH,
    max_length=MAX_TARGET_LENGTH,
)