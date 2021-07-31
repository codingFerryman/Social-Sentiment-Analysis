from collections import Counter
from typing import Tuple
import sys
import os
import pandas as pd
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.inputFunctions import loadData
from utils import get_project_path

data_pos, data_neg, test_full = loadData(ratio='full')
PROJECT_DIRECTORY = get_project_path()

def extract_hashtag(text_list: list) -> Tuple[list, int]:
    hashtag_list = []
    sentence_count = 0
    for text in text_list:
        for word in text.split():
            if word.startswith('#'):
                hashtag = word[1:]
                if len(hashtag) != 0:
                    if '#' in hashtag:
                        hashtag = hashtag.replace('#', '')
                    hashtag_list.append(hashtag)
                    sentence_count += 1
    return hashtag_list, sentence_count


hashtag_pos, hashtag_pos_sentence_count = extract_hashtag(data_pos)
hashtag_neg, hashtag_neg_sentence_count = extract_hashtag(data_neg)
hashtag_test, hashtag_test_sentence_count = extract_hashtag(test_full)

hashtag_pos_count = Counter(hashtag_pos)
hashtag_pos_count = sorted(hashtag_pos_count.items(), key=lambda x: x[1], reverse=True)
hashtag_neg_count = Counter(hashtag_neg)
hashtag_neg_count = sorted(hashtag_neg_count.items(), key=lambda x: x[1], reverse=True)
hashtag_test_count = Counter(hashtag_test)
hashtag_test_count = sorted(hashtag_test_count.items(), key=lambda x: x[1], reverse=True)


pos_hashtag_ratio = 100 * hashtag_pos_sentence_count / len(data_pos)
neg_hashtag_ratio = 100 * hashtag_neg_sentence_count / len(data_neg)
test_hashtag_ratio = 100 * hashtag_test_sentence_count / len(test_full)


pos_df = pd.DataFrame(hashtag_pos_count, columns=['Hashtag', 'PosFreq']).set_index('Hashtag')
neg_df = pd.DataFrame(hashtag_neg_count, columns=['Hashtag', 'NegFreq']).set_index('Hashtag')


full_df = pos_df.join(neg_df, how='outer').fillna(0)
full_df['PosRatio'] = full_df['PosFreq'] / (full_df['PosFreq'] + full_df['NegFreq'])
full_df['NegRatio'] = 1 - full_df['PosRatio']

full_df_highfreq = full_df[(full_df.PosFreq > 1000) | (full_df.NegFreq > 1000)]

full_df_highfreq_high_ratio = full_df_highfreq[(full_df_highfreq.PosRatio > 0.7) | (full_df_highfreq.NegRatio > 0.7)]

with open(Path(PROJECT_DIRECTORY, 'src', 'models', 'hashtag.json'), 'w') as fp:
    fp.write(full_df.to_json(orient='index', indent=4))

with open(Path(PROJECT_DIRECTORY, 'src', 'models', 'hashtag_highfreq_highratio.json'), 'w') as fp:
    fp.write(full_df_highfreq_high_ratio.to_json(orient='index', indent=4))
