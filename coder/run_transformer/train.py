# import sys
# sys.path.append('../../')
# sys.path.append('/ssd1/liboran/gitspace/github_Librarvl/LLM-experiment/')
# print(sys.path)

from transformers_s.src.transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers_s.src.transformers.models.gpt2.modeling_gpt2 import GPT2Model
import transformers_s.src.transformers.models.deprecated.deta

# from transformers import AutoTokenizer, AutoModel



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
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)

    def load_model(self, args):
        # self.model = AutoModel.from_pretrained(self.model_path)
        self.model = GPT2Model.from_pretrained(self.model_path)
        self.model.cuda()

    def test_model(self, args):
        text = "Replace me by any text you'd like."
        encoded_input = self.tokenizer(text, return_tensors='pt').to('cuda')
        output = self.model(**encoded_input)
        # self.tokenizer.decode(output[0].argmax(dim=-1))

        print(output)

