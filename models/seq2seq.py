"""The proposed model: BiLSTM encoder + Bahdanau attention + LSTM decoder."""
import keras

from models.decoder import AttentionDecoder
from models.encoder import BiLSTMEncoder


class Seq2SeqAttention(keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, attn_units=128, **kwargs):
        super().__init__(**kwargs)
        self.supports_jit = False   # the step-by-step decoder loop is not XLA-friendly
        self.encoder = BiLSTMEncoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_dim, attn_units)

    def call(self, inputs, return_attention=False, training=None):
        enc_ids, dec_ids = inputs
        enc_out, h, c, enc_mask = self.encoder(enc_ids)
        probs, attn = self.decoder(dec_ids, enc_out, enc_mask, h, c)
        return (probs, attn) if return_attention else probs
    