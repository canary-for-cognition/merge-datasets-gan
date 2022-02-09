from torch import nn, Tensor


class Encoder(nn.Module):

    def __init__(self, modality, sequence_length, batch_size, number_of_features, embedding_dim):
        super(Encoder, self).__init__()
        # def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        if modality == 'sequences':
            # self.n_layers = n_layers
            # self.hidden_size = hidden_sizes
            # self.embedding = embedding
            # # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
            # #   because our input size is a word embedding with number of features == hidden_size
            # self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
            #                 dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
            self.batch_size = batch_size
            self.modality = modality
            self.seq_len, self.n_features = sequence_length, number_of_features
            self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

            self.rnn1 = nn.LSTM(
            input_size=number_of_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )

            self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
            )
        elif modality == 'images':
            # for scanpaths generated from eye-tracking sequences
            pass
        elif modality == 'text':
            pass

    def forward(self, x):
        # def forward(self, input_seq, input_lengths, hidden=None):
        if self.modality == 'sequences':
            # # Convert input_seq to embeddings
            # embedded = self.embedding(input_seq)
            # # Pack padded batch of sequences for RNN module
            # packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
            # # Forward pass through GRU
            # outputs, hidden = self.gru(packed, hidden)
            # # Unpack padding
            # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            # # Sum bidirectional GRU outputs
            # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
            # # Return output and final hidden state
            # return outputs, hidden

            # print(x.shape)
            # Tutorial: https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
            # There're errors in code, so I followed the code change suggested by one discussion: Niki below.
            # x = x.reshape((1, self.seq_len, self.n_features))
            x = x.reshape((self.batch_size, self.seq_len, self.n_features))
            x, (_, _) = self.rnn1(x)
            x, (hidden_n, _) = self.rnn2(x)
            # print(hidden_n.shape)
            # return hidden_n.reshape((self.n_features, self.embedding_dim))
            return hidden_n.reshape((self.batch_size, self.embedding_dim))
        elif self.modality == 'images':
            pass
        elif self.modality == 'text':
            pass
