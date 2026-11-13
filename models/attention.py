"""Bahdanau (additive) attention, implemented from scratch."""
from keras import layers, ops


class BahdanauAttention(layers.Layer):
    """score(s, h_i) = v^T tanh(W1 h_i + W2 s)
    s = decoder state, h_i = encoder output at input position i."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units)
        self.v = layers.Dense(1, use_bias=False)

    def project_encoder(self, enc_out):
        return self.W1(enc_out)            # computed once, reused at every decoder step

    def call(self, query, enc_out, enc_proj, enc_mask):
        q = ops.expand_dims(self.W2(query), 1)                         # (batch, 1, units)
        scores = ops.squeeze(self.v(ops.tanh(enc_proj + q)), axis=-1)  # (batch, 500)
        scores = scores + (1.0 - enc_mask) * -1e9                      # ignore <pad>
        weights = ops.softmax(scores, axis=-1)                         # attention weights
        context = ops.sum(ops.expand_dims(weights, -1) * enc_out, axis=1)  # context vector
        return context, weights