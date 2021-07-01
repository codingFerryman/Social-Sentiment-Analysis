from collections import Counter
from typing import Tuple

import pandas as pd
from icecream import ic

from inputFunctions import loadData

ic.disable()

data_pos, data_neg, test_full = loadData(ratio='full')


def extract_hashtag(text_list: list) -> Tuple[list, int]:
    hashtag_list = []
    sentence_count = 0
    for text in text_list:
        for word in text.split():
            if word[0] == '#':
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
ic(hashtag_pos_sentence_count, hashtag_neg_sentence_count, hashtag_test_sentence_count)

hashtag_pos_count = Counter(hashtag_pos)
hashtag_pos_count = sorted(hashtag_pos_count.items(), key=lambda x: x[1], reverse=True)
hashtag_neg_count = Counter(hashtag_neg)
hashtag_neg_count = sorted(hashtag_neg_count.items(), key=lambda x: x[1], reverse=True)
hashtag_test_count = Counter(hashtag_test)
hashtag_test_count = sorted(hashtag_test_count.items(), key=lambda x: x[1], reverse=True)

ic(hashtag_pos_count[:3], hashtag_neg_count[:3], hashtag_test_count[:3])

pos_hashtag_ratio = 100 * hashtag_pos_sentence_count / len(data_pos)
neg_hashtag_ratio = 100 * hashtag_neg_sentence_count / len(data_neg)
test_hashtag_ratio = 100 * hashtag_test_sentence_count / len(test_full)

ic(pos_hashtag_ratio, neg_hashtag_ratio, test_hashtag_ratio)

pos_df = pd.DataFrame(hashtag_pos_count, columns=['Hashtag', 'PosFreq']).set_index('Hashtag')
neg_df = pd.DataFrame(hashtag_neg_count, columns=['Hashtag', 'NegFreq']).set_index('Hashtag')

ic(bool(set(pos_df.index) & set(neg_df.index)))

full_df = pos_df.join(neg_df, how='outer').fillna(0)
full_df['PosRatio'] = full_df['PosFreq'] / (full_df['PosFreq'] + full_df['NegFreq'])
full_df['NegRatio'] = 1 - full_df['PosRatio']
ic(full_df.describe())

full_df_highfreq = full_df[(full_df.PosFreq > 100) | (full_df.NegFreq > 100)]
ic(full_df_highfreq.describe())

full_df_highfreq_high_ratio = full_df_highfreq[(full_df_highfreq.PosRatio > 0.7) | (full_df_highfreq.NegRatio > 0.7)]
ic(full_df_highfreq_high_ratio.describe())

with open('hashtag.json', 'w') as fp:
    fp.write(full_df.to_json(orient='index', indent=4))

with open('hashtag_highfreq_high_ratio.json', 'w') as fp:
    fp.write(full_df_highfreq_high_ratio.to_json(orient='index', indent=4))

# test_matched_df = full_df_highfreq[full_df_highfreq.index.isin(hashtag_test)]
# ic(test_matched_df.describe())

# TODO: Slang?
