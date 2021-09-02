import json


def _loads_json(loadpath):
    with open(loadpath, 'r', encoding='utf-8') as fh:
        dataset = []
        for line in fh:
            example = json.loads(line)
            dataset.append(example)
    return dataset

def dumps_json(data, savepath):
    with open(savepath, 'w', encoding='utf-8') as fh:
        for example in data:
            fh.write(json.dumps(example) + '\n')

def _load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

train_qa_file="../data/sharc_raw/json/sharc_open_train.jsonl"
validation_qa_seen_file="../data/sharc_raw/json/sharc_open_dev_seen.jsonl"
validation_qa_unseen_file="../data/sharc_raw/json/sharc_open_dev_unseen.jsonl"
test_qa_seen_file="../data/sharc_raw/json/sharc_open_test_seen.jsonl"
test_qa_unseen_file="../data/sharc_raw/json/sharc_open_test_unseen.jsonl"
tree_mapping_file="./data/roberta-base-tree-mapping.json"

train_qa = _loads_json(train_qa_file)
dev_qa = _loads_json(validation_qa_seen_file) + _loads_json(validation_qa_unseen_file)
test_qa = _loads_json(test_qa_seen_file) + _loads_json(test_qa_unseen_file)
tree_mapping_data = _load_json(tree_mapping_file)

train_qa_treeids = set([_ex['tree_id'] for _ex in train_qa])
dev_qa_treeids = set([_ex['tree_id'] for _ex in dev_qa])
test_qa_treeids = set([_ex['tree_id'] for _ex in test_qa])

datasets = {}
for tree_id, m in tree_mapping_data.items():
    curr_output = []
    snippet = m['snippet']
    for q, clause in m['processed_snippet']['q2clause'].items():
        curr_output.append({
            'snippet': snippet,
            'span': clause['clause_span_text'],
            'question': q,
        })
    datasets[tree_id] = curr_output

train_seq2seq, dev_seq2seq, test_seq2seq = [], [], []
for tree_id in train_qa_treeids:
    train_seq2seq.extend(datasets[tree_id])
for tree_id in dev_qa_treeids:
    dev_seq2seq.extend(datasets[tree_id])
for tree_id in test_qa_treeids:
    test_seq2seq.extend(datasets[tree_id])

train_qg_file="../data/sharc_raw/json/sharc_open_question_generation_train.jsonl"
dev_qg_file="../data/sharc_raw/json/sharc_open_question_generation_dev.jsonl"
test_qg_file="../data/sharc_raw/json/sharc_open_question_generation_test.jsonl"
dumps_json(train_seq2seq, train_qg_file)
dumps_json(dev_seq2seq, dev_qg_file)
dumps_json(test_seq2seq, test_qg_file)

print('debug')