from torch import nn, Tensor


class Decoder(nn.Module):

    def __init__(self, modality, sequence_length, batch_size, number_of_features, input_dimension=128):
    # def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        if modality == 'sequences':
            # # Keep for reference
            # self.hidden_size = hidden_size
            # self.output_size = output_size
            # self.n_layers = n_layers
            # self.dropout = dropout

            # # Define layers
            # self.embedding = embedding
            # self.embedding_dropout = nn.Dropout(dropout)
            # self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
            self.batch_size = batch_size
            self.modality = modality
            self.seq_len, self.input_dim = sequence_length, input_dimension
            self.hidden_dim, self.n_features = 2 * input_dimension, number_of_features
            self.rnn1 = nn.LSTM(
            input_size=input_dimension,
            hidden_size=input_dimension,
            num_layers=1,
            batch_first=True
            )
            self.rnn2 = nn.LSTM(
            input_size=input_dimension,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
            self.output_layer = nn.Linear(self.hidden_dim, number_of_features)
        elif modality == 'images':
            # for scanpaths generated from eye-tracking sequences
            pass
        elif modality == 'text':
            pass

    def forward(self, x):
    # def forward(self, input_step, last_hidden, encoder_outputs):
        if self.modality == 'sequences':
            # # Note: we run this one step at a time
            # # Get embedding of current input
            # embedded = self.embedding(input_step)
            # embedded = self.embedding_dropout(embedded)
            # # Forward through unidirectional GRU
            # output, hidden = self.gru(embedded, last_hidden)
            # # Return output and final hidden state
            # return output, hidden

            # Tutorial: https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
            # There're errors in code, so I followed the code change suggested by one discussion: Niki below.
            # x = x.repeat(self.seq_len, self.n_features)
            x = x.repeat(self.seq_len, 1)
            # x = x.reshape((self.n_features, self.seq_len, self.input_dim))
            x = x.reshape((self.batch_size, self.seq_len, self.input_dim))
            x, (hidden_n, cell_n) = self.rnn1(x)
            x, (hidden_n, cell_n) = self.rnn2(x)
            # x = x.reshape((self.seq_len, self.hidden_dim))
            x = x.reshape((self.batch_size, self.seq_len, self.hidden_dim))
            return self.output_layer(x)
        elif self.modality == 'images':
            pass
        elif self.modality == 'text':
            pass
