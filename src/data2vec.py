#Python libraries
import datetime
import random
import string

#Third Party libraries
import numpy as np


class vectorizeData:
    '''
    This class transforms the text data into a vector that
    would be used as an input to our scRNN model.
    Structure of the vector: [Beg,Int,End]
    '''
    def __init(self, TrainPath, TestPath, batchSize, jumble_type = "NO", noise_type = "INSERT"):
        '''
        inp:
        TrainPath -> Path of training data
        TestPath -> Path of testing data
        batchSize -> Size of the batch in which data will be divided
        noise_type -> Type of noise to be added in train data
        jumble_type -> Reordering of train data
        '''
        self.TrainPath = TrainPath
        self.TestPath = TestPath
        self.batchSize = batchSize
        self.alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"
        self.jumble_type = jumble_type

    def loadData(self, path):
        '''
        Load the data from the path
        '''
        self.vocab = {}
        self.idx2vocab = {}
        words = open(path).read().replace('\n', '<eos>')
        self.wordsDataset = np.ndarray((len(words),), dtype=np.int32)

        for i,word in enumerate(words):
            #Generate one hot encoding for every word
            if word not in vocab:
                self.vocab[word] = len(self.vocab)
                self.idx2vocab[self.vocab[word]] = word

            self.wordsDataset[i] = vocab[word]

    def loadVectors(self):
        self.trainData = self.loadData(self.TrainPath)
        self.trainCleaned = open(TrainPath).read().replace('\n', '<eos>')

    def vectorizeData(self):
        XVec = np.zeros(int(len(self.trainCleaned)/self.batchSize), self.batchSize, len(self.alph))
        YVec = np.zeros(int(len(self.trainCleaned)/self.batchSize), self.batchSize, len(self.vocab))
        X_token = []

        for i, miniBatchTokens in enumerate(zip(*iter()*self.batchSize)):
            X_token_m = []
            xMiniBatch = np.zeros((self.batchSize, len(self.alph)*3))
            yMiniBatch = np.zeros((self.batchSize), len(self.vocab))

            for j,token in enumerate(miniBatchTokens):
                if jumble_type = 'NO':
                    xMiniBatch[j], x_token = self.addNoise(token, noise_type)
                else:
                    xMiniBatch[j], x_token = self.jumbleChar(token, jumble_type)

                bin_label = [0] * len(vocab)
                bin_label[vocab[word]] = 1
                yMiniBatch[j] = np.array(bin_label)
                X_token_m.append(X_token)

            XVec[m] = xMiniBatch
            YVec[m] = yMiniBatch
            X_token.append(X_token_m)


            percentage = int(m*100. / (len(self.trainCleaned) / self.batchSize))
            sys.stdout.write("\r%d %% %s" % (percentage, data_name))
            sys.stdout.flush()

        print()
        return X_vec, Y_vec, X_token


    def hasNum(self, word):
        for char in word:
            if char.isdigit():
                return True

            return False

    def addNoise(self, word, noise_type):
        if noise_type == "DELETE":
            bin_all = [0] * len(self.alph)
            if word = '<eos>':
                bin_all[-1] += 1
            elif word = '<unk>':
                bin_all[-2] +=1
            elif self.hasNum(word):
                bin_all[-3] += 1
            else:
                idx = random.randint(0, len(w)-1)
                w = w[:idx] + w[idx+1:]
                for i in range(len(w)):
                    try:
                        bin_all[self.alph.index(w[i])] +=1
                    except ValueError:
                        print(w)
                        raise

            return np.array(bin_all * 3), w


        if noise_type == "INSERT":
            bin_all = [0] * len(self.alph)
            if word = '<eos>':
                bin_all[-1] += 1
            elif word = '<unk>':
                bin_all[-2] +=1
            elif self.hasNum(word):
                bin_all[-3] += 1
            else:
                ins_idx = random.randint(0, len(w)-1)
                ins_char_idx = np.random.randint(0, len(string.ascii_lowercase))
                ins_char = list(string.ascii_lowercase)[ins_char_idx]
                w = w[:ins_idx] + ins_char + w[ins_idx:]

                for i in range(len(w)):
                    try:
                        bin_all[self.alph.index(w[i])] += 1
                    except:
                        print(w)
                        raise

            return np.array(bin_all * 3), w


        if noise_type == "REPLACE"
            bin_all = [0] * len(self.alph)
            if word = '<eos>':
                bin_all[-1] += 1
            elif word = '<unk>':
                bin_all[-2] +=1
            elif self.hasNum(word):
                bin_all[-3] += 1
            else:
                rep_idx = random.randint(0, len(w)-1)
                rep_char_idx = np.random.randint(0, len(string.ascii_lowercase))
                rep_char = list(string.ascii_lowercase)[rep_char_idx]
                w = w[:rep_idx] + rep_char + w[rep_idx+1:]

                for i in range(len(w)):
                    try:
                        bin_all[i] +=1
                    except ValueError:
                        print(w)
                        raise

            return np.array(bin_all * 3), w


    def jumbleChar(self, word, jumble_type):
        if jumble_type == "WHOLE":
            bin_all = [0] * len(self.alph)
            bin_filler = [0] * (len(self.alph) * 2)
            if word = '<eos>':
                bin_all[-1] += 1
            elif word = '<unk>':
                bin_all[-2] +=1
            elif self.hasNum(word):
                bin_all[-3] += 1
            else:
                w = ''.join(random.sample(w, len(w)))
                for i in range(len(w)):
                    try:
                        bin_all[self.alph.index(w[i])] += 1
                    except ValueError:
                        print(w)
                        raise

            bin_all += bin_filler
            return np.array(bin_all), w

        if jumble_type == "BEG":
            bin_initial = [0]*len(alph)
            bin_end = [0]*len(alph)
            bin_filler = [0]*len(alph)
            if w == '<eos>':
                bin_initial[-1] += 1
                bin_end[-1] += 1
            elif w == '<unk>':
                bin_initial[-2] += 1
                bin_end[-2] += 1
            elif self.hasnum(w):
                bin_initial[-3] += 1
                bin_end[-3] += 1
            else:
                if len(word) > 3:
                    w_init = ''.join(random.sample(w[:-1], len(w[:-1])))
                    w = w_init + w[-1]

                for i in range(len(w)):
                    try:
                        if i==len(w)-1:
                            bin_end[self.alph.index(w[i]) += 1
                        else:
                            bin_initial[i] +=1

                    except ValueError:
                        print(w)
                        raise

            bin_all = bin_initial + bin_end + bin_filler
            return np.array(bin_all), w


        elif opt == "END":
            bin_initial = [0]*len(alph)
            bin_end = [0]*len(alph)
            bin_filler = [0]*len(alph)
            if w == '<eos>':
                bin_initial[-1] += 1
                bin_end[-1] += 1
            elif w == '<unk>':
                bin_initial[-2] += 1
                bin_end[-2] += 1
            elif self.hasnum(w):
                bin_initial[-3] += 1
                bin_end[-3] += 1
            else:
                if len(w) > 3:
                    w_end = ''.join(random.sample(w[1:], len(w[1:])))
                    w = w[0] + w_end

                for i in range(len(w)):
                    try:
                        if i==0:
                            bin_initial[alph.index(w[i])] += 1
                        else:
                            bin_end[alph.index(w[i])] += 1
                    except ValueError:
                        print(w)
                        raise
            bin_all = bin_initial + bin_end + bin_filler
            return np.array(bin_all), w

        elif opt == "INT":
            bin_initial = [0]*len(alph)
            bin_middle = [0]*len(alph)
            bin_end = [0]*len(alph)
            if w == '<eos>':
                bin_initial[-1] += 1
                bin_middle[-1] += 1
                bin_end[-1] += 1
            elif w == '<unk>':
                bin_initial[-2] += 1
                bin_middle[-2] += 1
                bin_end[-2] += 1
            elif self.hasnum(w):
                bin_initial[-3] += 1
                bin_middle[-3] += 1
                bin_end[-3] += 1
            else:
                if len(w) > 3:
                    w_mid = ''.join(random.sample(w[1:-1], len(w[1:-1])))
                    w = w[0] + w_mid + w[-1]
                for i in range(len(w)):
                    try:
                        if i==0:
                            bin_initial[alph.index(w[i])] += 1
                        elif i==len(w)-1:
                            bin_end[alph.index(w[i])] += 1
                        else:
                            bin_middle[alph.index(w[i])] += 1
                    except ValueError:
                        print(w)
                        raise
            bin_all = bin_initial + bin_middle + bin_end
            return np.array(bin_all), w


        else:
            raise


if __name__ == "__main__":
    trainPath = './data/ptb.train.txt'
    testPath = './data/ptb.test.txt'
    batchSize = 20
    data = vectorizeData(trainPath, testPath, batchSize)
