"""Shared training, loss and decoding code (same for every model)."""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.tokenizer import EOS_ID, PAD_ID, SOS_ID


def masked_loss(y_true, y_pred):
    """Cross-entropy that ignores <pad> positions (so padding can't be 'learned')."""
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), loss.dtype)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)


def load_arrays(name):
    d = np.load(f"dataset/processed/{name}_arrays.npz")
    return d["encoder_input"], d["decoder_input"], d["decoder_target"]


def train_model(model, name, save_dir, epochs, batch_size, lr, patience):
    os.makedirs(save_dir, exist_ok=True)
    tr_e, tr_di, tr_dt = load_arrays("train")
    va_e, va_di, va_dt = load_arrays("val")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=masked_loss,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(
            f"{save_dir}/{name}.weights.h5",
            monitor="val_loss", save_best_only=True, save_weights_only=True,
        ),
    ]
    history = model.fit(
        [tr_e, tr_di], tr_dt,
        validation_data=([va_e, va_di], va_dt),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
    )
    with open(f"{save_dir}/{name}_history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
    return history


def greedy_decode(model, tok, enc_ids, summary_len):
    """Generate a summary word by word, always picking the most likely next word."""
    dec_len = summary_len - 1
    dec = np.zeros((1, dec_len), dtype="int32")
    dec[0, 0] = SOS_ID
    enc = enc_ids[None, :]
    for t in range(dec_len):
        probs = model([enc, dec], training=False).numpy()[0, t]
        probs[[PAD_ID, SOS_ID]] = 0
        nxt = int(np.argmax(probs))
        if nxt == EOS_ID:
            break
        if t + 1 < dec_len:
            dec[0, t + 1] = nxt
    return tok.decode(dec[0])