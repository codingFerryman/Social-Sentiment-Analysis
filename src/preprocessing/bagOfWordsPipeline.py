import tensorflow as tf
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inputFunctions
import loggers

logger = loggers.getLogger("BagOfWordsPipeline", debug=True)

class BagOfWordsPipeLine:
    def __init__(self, dataPath=None, loadFunction:callable=None):
        logger.info("BagOfWordsPipeline created")
        self.dataPath = dataPath
        self._tokenizer = None
        self.allData = []
        self.dataPos = []
        self.dataNeg = []
        if loadFunction == None:
            self.loadFunction = inputFunctions.loadData
        else:
            self.loadFunction = loadFunction
    
    def loadData(self):
        train_pos, train_neg, test_data = self.loadFunction(self.dataPath)
        self.dataPos = train_pos
        self.dataNeg = train_neg
        self.allData = train_pos + train_neg

    def trainTokenizer(self):
        logger.info("Creating bag of words tokenizer")
        self._tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None,
                                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                lower=True, split=' ', char_level=False, oov_token=None,
                                document_count=0)
        logger.info("Starting training of bag of words tokenizer")
        self._tokenizer.fit_on_texts(self.allData)
        logger.info("Finished training of bag of words tokenizer")
    
    def textsToSequences(self, texts: list) -> tf.Tensor:
        return self._tokenizer.texts_to_sequences(texts)
        
    def textsToPaddedSequences(self, texts: list):
        logger.info("transforming texts to padded sequences with bag of words tokenizer")
        sequences = self.textsToSequences(texts)
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    
    def textsToMatrix(self, texts: list) -> tf.Tensor:
        logger.info("transforming texts to matrix with bag of words tokenizer")
        sequencesPos = self.textsToSequences(texts) 
        # paddedSequences = tf.keras.preprocessing.sequence.pad_sequences(sequencesPos, padding='post')
        return self._tokenizer.sequences_to_matrix(sequencesPos)


    def textsPosToPaddedSequences(self):
        return self.textsToPaddedSequences(self.dataPos)

    def textsNegToPaddedSequences(self):
        return self.textsToPaddedSequences(self.dataNeg)

    def textsPosToMatrix(self):
        return self.textsToMatrix(self.dataPos)

    def textsNegToMatrix(self):
        return self.textsToMatrix(self.dataNeg)