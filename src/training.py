import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import RNN_TextClassification
from preprocess import Preprocess
import torch.optim as optim
import torch.nn.functional as F
from parser_param import parameter_parser

SEED = 2019
torch.manual_seed(2019)

class DataMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) :
        return self.x[index], self.y[index]

class Execute:
    def __init__(self, args):
        self.__init_data__(args)

        self.batch_size = args.batch_size
        self.model = RNN_TextClassification(args)

    def __init_data__(self, args):
        self.preprocess = Preprocess(args)
        self.preprocess.load_data()
        self.preprocess.Tokenization()

        raw_train_data = self.preprocess.X_train
        raw_test_data = self.preprocess.X_test

        self.y_train = self.preprocess.Y_train
        self.y_test = self.preprocess.Y_test

        self.x_train = self.preprocess.sequence_to_token(raw_train_data)
        self.x_test = self.preprocess.sequence_to_token(raw_test_data)

    def train(self):
        train = DataMapper(self.x_train, self.y_train)
        test = DataMapper(self.x_test, self.y_test)

        self.load_train = DataLoader(train, batch_size=self.batch_size)
        self.load_test = DataLoader(test)

        optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        print(self.model)

        if torch.cuda.is_available():
            device = "gpu"
            print(device)
        else:
            device = "cpu"
            print(device)
        for epoch in range(args.epochs):

            prediction = []

            self.model.train()
            
            for x_batch, y_batch in self.load_train:
                
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor).unsqueeze(1)

                y_pred = self.model(x)

                loss = F.binary_cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                prediction.append(y_pred.squeeze().detach().numpy())
            
        test_prediction = self.evaluation()
        torch.save(self.model, "model/RNN_classification.ckpt")

        train_accuracy = self.calculate_accuracy(self.y_train, prediction)
        test_accuracy = self.calculate_accuracy(self.y_test, test_prediction)

        print("Epoch : %.5f, Loss : %.5f, Train Accuracy : %.5f, Test accuracy : %.5f" % (epoch + 1, loss.item(), train_accuracy, test_accuracy)) #.5f : lam tron den so thu 5 sau dau phay

    def evaluation(self):
        prediction = []
        self.model.eval()

        for x_batch, y_batch in self.load_test:
            with torch.no_grad():
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor)

                y_pred = self.model(x)

                prediction.append(y_pred.squeeze().detach().numpy())
            
        return prediction

    @staticmethod
    def calculate_accuracy(groud_truth, prediction):
        true_positive = 0
        true_negative = 0

        for x, y in zip(groud_truth, prediction):
            if y >= 0.5 and x == 1:
                true_positive += 1
            elif y < 0.5 and x == 0:
                true_negative += 1
        
        return (true_positive + true_negative) / groud_truth
        
if __name__ == "__main__":
    args = parameter_parser()
    output = Execute(args)
    output.train()

        
        