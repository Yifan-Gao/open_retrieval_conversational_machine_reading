import os
import json

split = 'dev_seen'

home_path = '/home/ubuntu/data/MyFusionInDecoderOut/open_retrieval_sharc_private'

with open(os.path.join(home_path, "data", "sharc_raw", "json", "sharc_open_id2snippet.json")) as f:
    id2snippet = json.load(f)
with open(os.path.join(home_path, "data", "sharc_raw", "json", "sharc_open_snippet2id.json")) as f:
    snippet2id = json.load(f)

with open(os.path.join(home_path, "data", "tfidf", "{}.json".format(split))) as f:
    qa_data = json.load(f)

output = []
for ex in qa_data:
    if ex['scenario'] != "":
        if ex['scenario'][-1] != '.':
            question = ex['scenario'] + ". " + ex['question']
        else:
            question = ex['scenario'] + " " + ex['question']
    else:
        question = ex['question']
    curr = {
        'dataset': split,
        'question': question,
        'answers': [snippet2id[ex['snippet']]],
        'positive_ctxs': [{
            'title': "",

        }],
        'negative_ctxs': [],
        'hard_negative_ctxs': [],
    }
