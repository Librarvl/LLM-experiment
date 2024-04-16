from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer, GPT2Model


class TransformerTrainer():
    def __init__(self):
        self.model_path = "/ssd1/share/liboran/gpt2"
        # self.model_path = "/ssd1/share/liboran/bert-base-uncased"

    def train(self):
        args = self.model_path
        
        self.prepare_work(args)

    def prepare_work(self, args=""):
        self.load_tokenizer(args)
        self.load_model(args)
        self.test_model(args)

    def load_tokenizer(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def load_model(self, args):
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.cuda()

    def test_model(self, args):
        text = "Replace me by any text you'd like."
        encoded_input = self.tokenizer(text, return_tensors='pt').to('cuda')
        output = self.model(**encoded_input)
        self.tokenizer.decode(output[0].argmax(dim=-1))

        print(output)

