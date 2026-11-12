"""Baseline Seq2Seq models WITHOUT attention: simple RNN, LSTM and GRU.

The encoder reads the paper; its final hidden state(s) initialize the decoder.
Only the recurrent cell type differs between the three models.
"""
from tensorflow.keras import layers, Model

CELLS = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}


def build_baseline(cell_type, vocab_size, embed_dim, hidden_dim, article_len, decoder_len):
    cell = CELLS[cell_type]

    # ---- Encoder ----
    enc_in = layers.Input(shape=(article_len,), name="encoder_input")
    # mask_zero=True makes the encoder ignore <pad> positions
    enc_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="enc_embedding")(enc_in)
    enc_result = cell(hidden_dim, return_state=True, name="encoder")(enc_emb)
    enc_states = list(enc_result[1:])  # RNN/GRU: [h]   LSTM: [h, c]

    # ---- Decoder (teacher forcing: receives the true previous words) ----
    dec_in = layers.Input(shape=(decoder_len,), name="decoder_input")
    dec_emb = layers.Embedding(vocab_size, embed_dim, name="dec_embedding")(dec_in)
    dec_seq = cell(hidden_dim, return_sequences=True, name="decoder")(
        dec_emb, initial_state=enc_states
    )

    # ---- Output: probability of every vocabulary word at every step ----
    probs = layers.Dense(vocab_size, activation="softmax", name="output")(dec_seq)

    return Model([enc_in, dec_in], probs, name=f"{cell_type}_baseline")