import abc


class InputPipeline(metaclass=abc.ABCMeta):
    """ This is an interface that has to be implemented by
    classes that preprocess data before proviging them to models.
    For example you may want each word to be presented as a numerical value
    and then clear bad characters like "<" or ">" etc.
    After that the result can be provided to a model via textsToSequences method
    or via the textsToPaddedSequences.
    The method textsToMatrix requires TBs of memory typically, so it is unpracticall.
    """    
    @classmethod
    def __subclasshook__(self, subclass) -> bool:
        clsMethods = [
            f for f in dir(self)
            if callable(getattr(self, f)) and not f.startswith("__")
        ]
        subclassMethods = [
            f for f in dir(subclass)
            if callable(getattr(subclass, f)) and not f.startswith("__")
        ]
        comp = lambda s: getattr(subclass, s).__code__.co_varnames == getattr(
            clsMethods, s).__code__.co_varnames
        return all([(s in clsMethods and comp(s)) for s in subclassMethods])

    @abc.abstractmethod
    def loadData(self):
        """ This function loads data of positive and negative tweets from any specific 
            file/source provided in the constructor.

        """
        raise NotImplementedError
    # @abc.abstractmethod
    # def trainTokenizer(self):
    #     """ This function trains the tokenizer on the text that was provided.
    #     Essentially the tokenizer creates an encoding (in int) of each word appearing in the provided texts.
    #     """
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def textsToSequences(self, texts: list):
    #     """ This transforms the encoded words at each list candidate (tweet) to a tuple of encoded words (ints).
    #     Args:
    #         texts (list): list of texts (instances) with words to be encoded into int.
    #     Returns:
    #         tf.Tensor: the list of tuples with encoded words as a tensorflow tensor
    #     """
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def textsToPaddedSequences(self, texts: list):
    #     """ This method does the same thing as `textToSequences` but also pads the sequences.
    #     The sequences are padded with 0 at the beginning so that they have the same length.
    #
    #     Args:
    #         texts (list): list of texts (instances) with words to be encoded into int.
    #
    #     Returns:
    #         tf.Tensor: the list of padded tuples with encoded words as a tensorflow tensor
    #
    #     """
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def textsToMatrix(self, texts: list) -> 'tf.Tensor':
    #     """ This method transforms the encoded words at each list candidate (tweet) to
    #     a matrix with columns refering to words and cells carrying number of occurrence or presence.
    #     Typically needs too much memory and is used in direct application of ML algorithms
    #     (logistic regression, decision tree, etc.)
    #
    #     Args:
    #         texts (list): list of texts (instances) with words to be encoded into int.
    #
    #     Returns:
    #         tf.Tensor: The matrix with the converted texts.
    #     """
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def textsPosToPaddedSequences(self):
    #     """ positive tweets to padded sequences
    #
    #     Raises:
    #         NotImplementedError: [description]
    #     """
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def textsNegToPaddedSequences(self):
    #     """ negative tweets to padded sequences
    #
    #     Raises:
    #         NotImplementedError: [description]
    #     """
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def textsPosToMatrix(self) -> 'tf.Tensor':
    #     """ This method transforms the texts of positive tweets via the textsToMatrix method.
    #
    #     Returns:
    #         tf.Tensor: Texts with negative tweets to matrix tensor
    #     """
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def textsNegToMatrix(self) -> 'tf.Tensor':
    #     """ This method transforms the texts of negative tweets via the textsToMatrix method.
    #
    #     Returns:
    #         tf.Tensor: Texts with negative tweets to matrix tensor
    #     """
    #     raise NotImplementedError
