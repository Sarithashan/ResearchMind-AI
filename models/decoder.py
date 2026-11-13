"""LSTM decoder with Bahdanau attention (custom)."""
from keras import layers, ops

from models.attention import BahdanauAttention


class AttentionDecoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_dim, attn_units=128, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(attn_units)
        self.cell = layers.LSTMCell(hidden_dim)
        self.output_layer = layers.Dense(vocab_size, activation="softmax")

    def call(self, dec_ids, enc_out, enc_mask, init_h, init_c):
        emb = self.embedding(dec_ids)                    # teacher forcing: true previous words
        enc_proj = self.attention.project_encoder(enc_out)
        h, c = init_h, init_c
        outs, ctxs, attns = [], [], []
        for t in range(emb.shape[1]):
            # 1) attend using the PREVIOUS decoder state (original Bahdanau formulation)
            context, weights = self.attention(h, enc_out, enc_proj, enc_mask)
            # 2) feed [word embedding ; context vector] to the LSTM
            step_in = ops.concatenate([emb[:, t, :], context], axis=-1)
            out, (h, c) = self.cell(step_in, [h, c])
            outs.append(out); ctxs.append(context); attns.append(weights)
        features = ops.concatenate([ops.stack(outs, axis=1), ops.stack(ctxs, axis=1)], axis=-1)
        return self.output_layer(features), ops.stack(attns, axis=1)