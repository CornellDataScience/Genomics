from keras import backend as K
from keras.models import Sequential, Model, Input
from keras.layers import GRU,  Dense


class encoder:

    def __init__(self, encoding_size):
        inputs = Input(shape=(4, None))
        _, state_c = GRU(
            encoding_size,
            return_sequences=False,
            return_state=True)(inputs)

        self.encoder = Model(inputs=inputs, outputs=state_c)

        arange = K.linspace(0, K.exp(1), K.shape(inputs)[1])
        outseq = GRU(
            encoding_size,
            return_sequences=True,
            return_state=False,
            initial_state=state_c,
            activation="softmax")(arange)

        reconstruction = Dense((4, None))(outseq)

        self.autoencoder = Model(inputs=inputs, outputs=reconstruction)

        decoding_in = Input(shape=(encoding_size,))
        self.decoder = Model(decoding_in, self.autoencoder.layers[-2:])

        self.autoencoder.compile(optimizer="rmsprop",
                             loss="kullback_leibler_divergence",
                             metrics=['accuracy'])

    def fit(self, gene_train, gene_val, epochs=50, batch_size=128):
        self.autoencoder.fit(gene_train, gene_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             validation_data=(gene_val, gene_val))

    def encode(self, genes):
        return self.encoder()

    def save(self):
        pass

    def load(self):
        pass
