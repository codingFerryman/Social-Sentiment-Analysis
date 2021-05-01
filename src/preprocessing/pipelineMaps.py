import transformers
import typing

def getCTRLTokenizer() -> transformers.CTRLTokenizer:
    tokenizer = transformers.CTRLTokenizer.from_pretrained('ctrl')
    tokenizer.pad_token = ' '
    return tokenizer

def getGPT2Tokenizer() -> transformers.GPT2Tokenizer:
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = ' '
    return tokenizer

def getTransfoXLTokenizer() -> transformers.TransfoXLTokenizer:
    tokenizer = transformers.TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    tokenizer.pad_token = ' '
    return tokenizer
    
def getTokenizerMap() -> typing.Dict[str, typing.Callable]:
    return {
        "roberta": lambda: transformers.RobertaTokenizer.from_pretrained('roberta-base'),
        "electra": lambda: transformers.ElectraTokenizer.from_pretrained('google/electra-small-discriminator'),
        "albert" : lambda: transformers.AlbertTokenizer.from_pretrained('albert-base-v2'),
        "bart" : lambda: transformers.BartTokenizer.from_pretrained('facebook/bart-large'),
        "berttweet" : lambda: transformers.BertweetTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False), # accepts only inference (no retraining)
        # "bigbird" : lambda: transformers.BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base'),
        "convbert": lambda: transformers.ConvBertTokenizer.from_pretrained('bert-base-uncased'),
        "ctrl": getCTRLTokenizer,
        "deberta": lambda: transformers.DebertaTokenizer.from_pretrained('microsoft/deberta-base'),
        "debertav2": lambda: transformers.DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge'),
        "distilbert": lambda: transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        "funnel": lambda: transformers.FunnelTokenizer.from_pretrained('funnel-transformer/small-base'),
        # "ibert": lambda: transformers.RobertaTokenizer.from_pretrained('kssteven/ibert-roberta-base'),
        "mpnet": lambda: transformers.MPNetTokenizer.from_pretrained('microsoft/mpnet-base'),
        # "gpt": lambda: transformers.OpenAIGPTTokenizer.from_pretrained('openai-gpt'), # type object 'EnglishDefaults' has no attribute 'create_tokenizer'
        "gpt2": getGPT2Tokenizer,
        "squeezebert": lambda: transformers.SqueezeBertTokenizer.from_pretrained('squeezebert/squeezebert-uncased'),
        "transformerxl": getTransfoXLTokenizer,
        "xlm-mlm": lambda: transformers.XLMTokenizer.from_pretrained('xlm-mlm-en-2048'),
        "xlnet": lambda: transformers.XLNetTokenizer.from_pretrained('xlnet-large-cased')
    }

def mapStrToTransformersTokenizer(tokenizerName: str) -> transformers.PreTrainedTokenizer:
    tokenizerMap = getTokenizerMap()
    return tokenizerMap[tokenizerName]()

def getModelMapAvailableNames() -> typing.List[str]:
    return [k for k in getTokenizerMap().keys()]