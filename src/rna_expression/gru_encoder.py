from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import GRU,  Dense
from keras.preprocessing.text import one_hot


class GeneEncoder:

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

        self.history = None

    def fit(self, gene_train, gene_val, epochs=50, batch_size=128):
        self.history = self.autoencoder.fit(gene_train, gene_train,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(gene_val, gene_val))

    def encode(self, genes, batch_size=32):
        return self.encoder.predict(genes, batch_size)

    def save(self):
        self.encoder.save('encoder.h5')
        self.decoder.save('decoder.h5')
        self.autoencoder.save('autoencoder.h5')

    def load(self):
        self.encoder = load_model('encoder.h5')
        self.decoder = load_model('decoder.h5')
        self.autoencoder = load_model('autoencoder.h5')

    @staticmethod
    def gene_preprocess(gene, genecat=""):
        gen_code = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
        gene += genecat
        ret = K.zeros((4, len(gene)))
        for i, base in enumerate(gene):
            if base == 'N':
                continue
            ret[gen_code[base], i] = 1
        return ret

    def history_display(self):
        print(self.history.history)
