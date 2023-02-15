import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tm
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import BertConfig, BertLMHeadModel
from transformers import XLNetConfig, XLNetLMHeadModel
from transformers import BartConfig, BartModel
from transformers import OPTConfig, OPTModel
from transformers import AlbertConfig, AlbertModel
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


# non model data >> model data
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



class NoLeafModule(nn.Module):
    """
    In this no-leaf module, it has subordinate nn.modules and a nn.Parameter.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.proj1 = nn.Linear(1024, 2048)
        self.weight1 = nn.Parameter(torch.randn(2048, 2048))
        self.proj1_2 = nn.Linear(2048, 512)

        self.proj2 = nn.Linear(512, 2048)
        self.weight2 = nn.Parameter(torch.randn(2048, 2048))
        self.proj2_2 = nn.Linear(2048, 1024)

        self.proj3 = nn.Linear(1024, 1024)
        self.weight3 = nn.Parameter(torch.randn(1024, 1024))
        self.proj3_2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.proj1(x)
        x = F.linear(x, self.weight1)
        x = self.proj1_2(x)

        x = self.proj2(x)
        x = F.linear(x, self.weight2)
        x = self.proj2_2(x)

        x = self.proj3(x)
        x = F.linear(x, self.weight3)
        x = self.proj3_2(x)

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


class MyBart(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=12, num_attention_heads=16, vocab_size=50265,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = BartModel(BartConfig(d_model=hidden_size, encoder_layers=num_layers, decoder_layers=num_layers,
                                          encoder_attention_heads=num_attention_heads,
                                          decoder_attention_heads=num_attention_heads,
                                          max_position_embeddings=1024,
                                          vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class MyXL(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, vocab_size=32000,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = XLNetLMHeadModel(XLNetConfig(d_model=hidden_size, n_layer=num_layers,
                                                n_head=num_attention_heads, vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class MyOPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = OPTModel(config=OPTConfig(hidden_size=512, num_hidden_layers=6, num_attention_heads=16))

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0]


class MyAlbert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AlbertModel(config=AlbertConfig(embedding_size=1024,
                            hidden_size=1024,
                            num_hidden_layers=12,
                            num_attention_heads=16,
                            intermediate_size=2048))

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                          attention_mask=attention_mask).pooler_output



def albert_data_gen(device="meta"):
    input_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    token_type_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    attention_mask = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    meta_args = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    return meta_args


def opt_data_gen(device="meta"):
    input_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    attention_mask = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
    return kwargs


def t5_data_gen(device="meta"):
    input_ids = torch.zeros((HF_BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64, device=device)
    kwargs = dict(input_ids=input_ids)
    return kwargs


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


@non_distributed_component_funcs.register(name='albert')
def get_albert_components():
    seq_len = 16
    batchSize = 16

    def albert_model_builder(checkpoint=False):
        model = MyAlbert()
        return model

    def albert_data_gen(device="meta"):
        input_ids = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        token_type_ids = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        attention_mask = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        kwargs = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return kwargs

    return albert_model_builder, albert_data_gen


@non_distributed_component_funcs.register(name='bart')
def get_bart_components():
    seq_len = 16
    batchSize = 16

    def bart_model_builder(checkpoint=False):
        model = MyBart()
        return model

    def bart_data_gen(device="meta"):
        input_ids = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        attention_mask = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return bart_model_builder, bart_data_gen


@non_distributed_component_funcs.register(name='xlnet')
def get_xlnet_components():
    seq_len = 16
    batchSize = 16

    def xlnet_model_builder(checkpoint=False):
        model = MyXL()
        return model

    def xlnet_data_gen(device="meta"):
        input_ids = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        attention_mask = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return xlnet_model_builder, xlnet_data_gen


@non_distributed_component_funcs.register(name='opt')
def get_opt_components():
    seq_len = 16
    batchSize = 16

    def opt_model_builder(checkpoint=False):
        model = MyOPT()
        return model

    def opt_data_gen(device="meta"):
        input_ids = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        attention_mask = torch.zeros((batchSize, seq_len), dtype=torch.int64, device=device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return opt_model_builder, opt_data_gen


@non_distributed_component_funcs.register(name='mlp')
def get_mlp_components():
    batchSize = 2048
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


@non_distributed_component_funcs.register(name='alexnet')
def get_alexnet_components():
    batchSize = 1

    def alexnet_model_builder(checkpoint=False):
        model = tm.alexnet()
        return model

    def alexnet_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 3, 224, 224, device=device)
        kwargs = dict(x=data)
        return kwargs

    return alexnet_model_builder, alexnet_data_gen


@non_distributed_component_funcs.register(name='vgg16')
def get_vgg16_components():
    batchSize = 64

    def vgg16_model_builder(checkpoint=False):
        model = tm.vgg16()
        return model

    def vgg16_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 3, 224, 224, device=device)
        kwargs = dict(x=data)
        return kwargs

    return vgg16_model_builder, vgg16_data_gen


@non_distributed_component_funcs.register(name='resnet18')
def get_resnet18_components():
    batchSize = 64

    def resnet18_model_builder(checkpoint=False):
        model = tm.resnet18()
        return model

    def resnet18_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 3, 224, 224, device=device)
        kwargs = dict(x=data)
        return kwargs

    return resnet18_model_builder, resnet18_data_gen


@non_distributed_component_funcs.register(name='no_leaf_model')
def get_no_leaf_module_components():
    batchSize = 8

    def no_leaf_model_builder(checkpoint=False):
        model = NoLeafModule(checkpoint=checkpoint)
        return model

    def no_leaf_module_data_gen(device="meta"):
        data = torch.rand(int(batchSize), 1024, device=device)
        kwargs = dict(x=data)
        return kwargs

    return no_leaf_model_builder, no_leaf_module_data_gen