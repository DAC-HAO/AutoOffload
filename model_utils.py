import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tm
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import BertConfig, BertLMHeadModel
from registry import non_distributed_component_funcs

HF_BATCH_SIZE = 8
TM_BATCH_SIZE = 64
SEQ_LENGTH = 16


class MLPModel(nn.Module):

    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        dim = 8192
        self.fc_list = nn.ModuleList()
        for iii in range(20):
            self.fc_list.append(nn.Linear(dim, dim))

    def forward(self, x):
        for fc in self.fc_list:
            x = fc(x)
        return x


class SimpleNet(nn.Module):

    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.embed = nn.Embedding(2048, 1024)
        self.proj1 = nn.Linear(1024, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.proj2 = nn.Linear(1024, 2048)
        self.ln2 = nn.LayerNorm(2048)
        self.classifier = nn.Linear(2048, 2048)

    def forward(self, x):
        x = self.embed(x)
        x = self.proj1(x)
        x = self.ln1(x)
        x = self.proj2(x)
        x = self.ln2(x)
        x = self.classifier(x)
        return x



class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]



class BertLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, vocab_size=30522,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = BertLMHeadModel(BertConfig(n_embd=hidden_size, num_hidden_layers=num_layers,
                                                n_head=num_attention_heads, max_position_embeddings=768,
                                                vocab_size=vocab_size))

        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


@non_distributed_component_funcs.register(name='bert')
def get_bert_components():
    vocab_size = 30522
    seq_len = 16
    batchSize = 16

    def bert_model_builder(checkpoint=False):
        model = BertLMModel(hidden_size=768, num_layers=12, num_attention_heads=12, vocab_size=vocab_size,
                            checkpoint=checkpoint)
        return model

    def bert_data_gen(device="meta"):
        input_ids = torch.randint(0, vocab_size, (batchSize, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return bert_model_builder, bert_data_gen


@non_distributed_component_funcs.register(name='gpt2')
def get_gpt2_components():
    vocab_size = 502
    seq_len = 8
    batchSize = 64

    def gpt2_model_builder(checkpoint=False):
        model = GPTLMModel(hidden_size=8192, num_layers=4, num_attention_heads=32, vocab_size=vocab_size,
                           checkpoint=checkpoint)
        return model

    def gpt2_data_gen(device="meta"):
        input_ids = torch.randint(0, vocab_size, (batchSize, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return gpt2_model_builder, gpt2_data_gen



@non_distributed_component_funcs.register(name='mlp')
def get_mlp_components():
    batchSize = 512
    def mlp_model_builder(checkpoint=False):
        model = MLPModel(checkpoint=checkpoint)
        return model

    def mlp_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 8192, device=device)
        kwargs = dict(x=data)
        return kwargs

    return mlp_model_builder, mlp_data_gen


@non_distributed_component_funcs.register(name='simplenet')
def get_simplenet_components():
    batchSize = 16
    seq_len = 32
    def simplenet_model_builder(checkpoint=False):
        model = SimpleNet(checkpoint=checkpoint)
        return model

    def simplenet_data_gen(device="meta"):
        input_ids = torch.randint(low=0, high=1024, size=(batchSize, seq_len), device=device)
        kwargs = dict(x=input_ids)
        return kwargs

    return simplenet_model_builder, simplenet_data_gen