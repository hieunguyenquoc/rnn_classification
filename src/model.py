import torch
import torch.nn as nn

class RNN_TextClassification(nn.Module):
    def __init__(self,args):
        super(RNN_TextClassification,self).__init__()

        self.args = args
        self.input_size = args.max_words
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.dropout = 0.2

        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.RNN = nn.RNN(input_size = self.hidden_dim, hidden_size = self.hidden_dim, num_layers = self.num_layer, batch_first = True)

        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim*2)
        self.fc2 = nn.Linear(in_features=self.hidden_dim*2, out_features=1, bias=True)

    def forward(self, x):

        h = torch.zeros((self.num_layer, x.size(0), self.hidden_dim))
        c = torch.zeros((self.num_layer, x.size(0), self.hidden_dim))

        torch.nn.init.xavier_normal(h)
        torch.nn.init.xavier_normal(c)

        output = self.embedding(x)

        output, (hidden, cell) = self.RNN(output, (h,c))

        output = self.dropout(output)

        output = torch.relu_(self.fc1(output[:,-1,:]))

        output = self.dropout(output)

        output = torch.sigmoid(self.fc2(output))

        return output