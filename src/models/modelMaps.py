import transformers
import typing

def getModelMap() -> typing.Dict[str, typing.Callable]:
    return {
        "roberta": lambda: transformers.TFRobertaForSequenceClassification.from_pretrained('roberta-base'),
        "electra": lambda: transformers.TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator'),
        "albert" : lambda: transformers.TFAlbertForSequenceClassification.from_pretrained('albert-base-v2'),
        "bart": lambda: transformers.BartForSequenceClassification.from_pretrained('facebook/bart-large'),
        "berttweet": lambda: transformers.AutoModel.from_pretrained("vinai/bertweet-base"),
        # "bigbird": lambda: transformers.BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base'),
        "convbert": lambda: transformers.TFConvBertForSequenceClassification.from_pretrained('bert-base-uncased'),
        "ctrl": lambda: transformers.TFCTRLForSequenceClassification.from_pretrained('ctrl'),
        "deberta": lambda: transformers.DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base'),
        "debertav2": lambda: transformers.DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge'),
        "distilbert": lambda: transformers.TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'),
        "funnel": lambda: transformers.TFFunnelForSequenceClassification.from_pretrained('funnel-transformer/small-base'),
        # "ibert": lambda: transformers.IBertForSequenceClassification.from_pretrained('kssteven/ibert-roberta-base'),
        "mpnet": lambda: transformers.TFMPNetForSequenceClassification.from_pretrained('microsoft/mpnet-base'),
        # "gpt": lambda: transformers.TFOpenAIGPTForSequenceClassification.from_pretrained('openai-gpt'), # type object 'EnglishDefaults' has no attribute 'create_tokenizer'
        "gpt2": lambda: transformers.TFGPT2ForSequenceClassification.from_pretrained("gpt2"),
        "squeezebert": lambda: transformers.SqueezeBertForSequenceClassification.from_pretrained('squeezebert/squeezebert-uncased'),
        "transformerxl": lambda: transformers.TransfoXLForSequenceClassification.from_pretrained('transfo-xl-wt103'),
        "xlm-mlm": lambda: transformers.TFXLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048'),
        "xlnet": lambda: transformers.TFXLNetForSequenceClassification.from_pretrained('xlnet-large-cased')
    }

def mapStrToTransformerModel(modelName:str) -> transformers.PreTrainedModel:
    modelMap = getModelMap()
    return modelMap[modelName]()

def getModelMapAvailableNames() -> typing.List[str]:
    return [k for k in getModelMap().keys()]

# denoisers = {
#     "mbart": 
# }