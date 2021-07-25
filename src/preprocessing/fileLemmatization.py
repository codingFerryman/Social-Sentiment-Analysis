import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

if Path().resolve().parts[1] == 'cluster':
    import stanza

    proxy = "http://proxy.ethz.ch:3128"
    os.environ['HTTP_PROXY'] = proxy
    os.environ['HTTPS_PROXY'] = proxy

    os.environ['STANZA_RESOURCES_DIR'] = os.path.join(os.getenv("SCRATCH"), '.cache/stanza_resources/')
    stanza.download(lang='en', model_dir=os.getenv('STANZA_RESOURCES_DIR'))
    nlp = stanza.Pipeline(lang='en', dir=os.getenv('STANZA_RESOURCES_DIR'), processors='tokenize,pos,lemma')

else:
    import stanza

    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')


def lemmatization(text):
    doc = nlp(text)
    _result = " ".join([word.lemma for sent in doc.sentences for word in sent.words])
    return _result


def main(argv):
    with open(argv[1], 'r') as fp:
        _data = fp.readlines()
        _data_processed = []
        for _t in tqdm(_data, dynamic_ncols=True, desc="Lemmatization", mininterval=30, maxinterval=60):
            # _data_processed.append(lemmatization(_t) + '\n')
            pass
    output_path = argv[1].split('.')[0] + '_lemma' + argv[1].split('.')[-1]
    with open(output_path, 'w') as fp:
        fp.writelines(_data_processed)


if __name__ == '__main__':
    main(sys.argv)
