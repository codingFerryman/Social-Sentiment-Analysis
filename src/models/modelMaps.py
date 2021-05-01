import transformers
import tensorflow as tf
import typing

def getGPT2Model() ->transformers.TFGPT2ForSequenceClassification:
    model = transformers.TFGPT2ForSequenceClassification.from_pretrained("gpt2")
    model.training = True
    return model

def getModelMap() -> typing.Dict[str, typing.Callable]:
    return {
        "roberta": lambda: transformers.TFRobertaForSequenceClassification.from_pretrained('roberta-base'), # OK
        "electra": lambda: transformers.TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator'), # OK
        "albert" : lambda: transformers.TFAlbertForSequenceClassification.from_pretrained('albert-base-v2'), # OK
        "bart": lambda: transformers.BartForSequenceClassification.from_pretrained('facebook/bart-large'), # OK
        # "berttweet": lambda: transformers.AutoModel.from_pretrained("vinai/bertweet-base"), # accepts only inference (no retraining)
        # "bigbird": lambda: transformers.BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base'), # NOT OK (I dont remember why)
        "convbert": lambda: transformers.TFConvBertForSequenceClassification.from_pretrained('bert-base-uncased'), # OK
        "ctrl": lambda: transformers.TFCTRLForSequenceClassification.from_pretrained('ctrl'), # OK (needs too much memory)
        "deberta": lambda: transformers.DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base'), # OK
        "debertav2": lambda: transformers.DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge'), # OK
        "distilbert": lambda: transformers.TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'), # OK
        "funnel": lambda: transformers.TFFunnelForSequenceClassification.from_pretrained('funnel-transformer/small-base'), # OK
        # "ibert": lambda: transformers.IBertForSequenceClassification.from_pretrained('kssteven/ibert-roberta-base'), # NOT OK
        "mpnet": lambda: transformers.TFMPNetForSequenceClassification.from_pretrained('microsoft/mpnet-base'), # OK
        "gpt": lambda: transformers.TFOpenAIGPTForSequenceClassification.from_pretrained('openai-gpt'), # type object 'EnglishDefaults' has no attribute 'create_tokenizer'
        "gpt2": getGPT2Model, # OK (WIP trains only with batch size = 64 and seems it trains very fast)
        "squeezebert": lambda: transformers.SqueezeBertForSequenceClassification.from_pretrained('squeezebert/squeezebert-uncased'), # OK
        "transformerxl": lambda: transformers.TransfoXLForSequenceClassification.from_pretrained('transfo-xl-wt103'), # OK (demands insane amount of memory)
        "xlm-mlm": lambda: transformers.TFXLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048'), # OK
        "xlnet": lambda: transformers.TFXLNetForSequenceClassification.from_pretrained('xlnet-large-cased') # OK
    }

def mapStrToTransformerModel(modelName:str) -> transformers.PreTrainedModel:
    modelMap = getModelMap()
    return modelMap[modelName]()

def getModelMapAvailableNames() -> typing.List[str]:
    return [k for k in getModelMap().keys()]

# denoisers = {
#     "mbart": 
# }