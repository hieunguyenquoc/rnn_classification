import torch
from preprocess import Preprocess
from model import RNN_TextClassification
from parser_param import parameter_parser
from fastapi import FastAPI
import uvicorn

class Inference:
    def __init__(self):
        args = parameter_parser()
        self.model = RNN_TextClassification(args)
        self.model = torch.load("model/RNN_classification.ckpt")
        self.model.eval()
        self.preprocess = Preprocess(args)
        self.preprocess.load_data()
        self.preprocess.Tokenization()

    def predict(self, input):
        input = self.preprocess.sequence_to_token(input)
        input_tensor = torch.from_numpy(input)
        input_final = input_tensor.type(torch.LongTensor)

        result = self.model(input_final)
        return result.detach().numpy() #return a new tensor without require_grad = True to change it to numpy

app = FastAPI()

@app.post("/prediction")
def prediction(sentence:str):
    result = Inference()
    if result.predict([sentence]) >= 0.5:
        return "positve" 
    else:
        return "negative"

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)



        