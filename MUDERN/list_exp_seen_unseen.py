import json
import csv

all_results = [[] for _ in range(20)]
lentail = "8.0"
ltrans = "4"

for idx, seed in enumerate(["27", "95", "19", "87", "11"]):
    path = f"./out/v9_roberta-base_Lentail{lentail}_Ltrans{ltrans}_seed{seed}"
    for split in ['dev_seen', 'test_seen', 'dev_unseen', 'test_unseen']:
        with open(path+f"/results_{split}_e2e.json") as f:
            result = json.load(f)
        if split == 'test_seen':
            output_idx = idx + 5
        elif split == 'dev_unseen':
            output_idx = idx + 10
        elif split == 'test_unseen':
            output_idx = idx + 15
        else:
            output_idx = idx
        all_results[output_idx].append(result["macro_accuracy"])
        all_results[output_idx].append(result["micro_accuracy"])
        all_results[output_idx].append(result['classwise_accuracy_yes'])
        all_results[output_idx].append(result['classwise_accuracy_no'])
        all_results[output_idx].append(result['classwise_accuracy_more'])
        all_results[output_idx].append("")
        all_results[output_idx].append("")
        all_results[output_idx].append("")
        all_results[output_idx].append("")
        all_results[output_idx].append(result['fscore_bleu_1'])
        all_results[output_idx].append(result['fscore_bleu_2'])
        all_results[output_idx].append(result['fscore_bleu_3'])
        all_results[output_idx].append(result['fscore_bleu_4'])

with open("/research/dept7/ik_grp/ablation_seen_unseen_results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(all_results)













