#Python libraries
import datetime
import random

#Third Party libraries
import numpy as np


class vectorizeData:
    '''
    This class transforms the text data into a vector that
    would be used as an input to our scRNN model.
    Structure of the vector: [Beg,Int,End]
    '''
    def __init(self, TrainPath, TestPath):
        '''
        inp:
        TrainPath -> Path of training data
        TestPath -> Path of testing data

        '''
        self.TrainPath = TrainPath
        self.TestPath = TestPath
        self.alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"


    def loadData(self, path):
        '''
        Load the data from the path
        '''
        words = open(path).read().replace('\n', '<eos>')
        wordsDataset = np.ndarray((len(words),), dtype=np.int32)

        for i,word in enumerate(words):
            #Generate one hot encoding for every word
            if word not in vocab:
                vocab[word] = len(vocab)
                idx2vocab[vocab[words]] = word
