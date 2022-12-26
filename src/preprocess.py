import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re

class Preprocess:
    def __init__(self, args):
        self.data_path = "data/train.csv"
        self.test_size = args.test_size
        self.num_words = args.max_words
        self.max_len = args.max_len

    def load_data(self):
        df = pd.read_csv(self.data_path)

        df = df.drop(['id','keyword','location'], axis=1)

        df_train = df['text'].values
        df_label = df['target'].values
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(df_train, df_label, test_size=self.test_size)

    def remove_special_character(self):
        self.X_train = [x.lower() for x in self.X_train]
        self.X_train = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.X_train]
    
    def Tokenization(self):
        self.token = Tokenizer(num_words=self.num_words)
        self.token.fit_on_texts(self.X_train)
    
    def sequence_to_token(self, input):
        sequence = self.token.texts_to_sequences(input)
        return pad_sequences(sequence, maxlen=self.max_len)
        
    
    
