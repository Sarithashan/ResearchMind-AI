"""Bidirectional LSTM encoder (custom)."""
from keras import layers, ops


class BiLSTMEncoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.bilstm = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        )
        # squeeze forward+backward final states into the decoder's initial state
        self.bridge_h = layers.Dense(hidden_dim, activation="tanh")
        self.bridge_c = layers.Dense(hidden_dim, activation="tanh")

    def call(self, enc_ids):
        x = self.embedding(enc_ids)
        outputs, fh, fc, bh, bc = self.bilstm(x)       # outputs: (batch, 500, 2*hidden)
        init_h = self.bridge_h(ops.concatenate([fh, bh], axis=-1))
        init_c = self.bridge_c(ops.concatenate([fc, bc], axis=-1))
        enc_mask = ops.cast(ops.not_equal(enc_ids, 0), "float32")  # 1 = real word, 0 = <pad>
        return outputs, init_h, init_c, enc_mask