import json
import csv

all_results = [[] for _ in range(10)]
lentail = "8.0"
ltrans = "4"

for idx, seed in enumerate(["27", "95", "19", "87", "11"]):
    path = f"./out/ablation_discern_like_roberta-base_Lentail{lentail}_Ltrans{ltrans}_seed{seed}"
    for split in ['dev', 'test']:
        with open(path+f"/results_{split}.json") as f:
            result = json.load(f)
        if split == 'test': idx += 5
        all_results[idx].append(result["eval_macro_accuracy"])
        all_results[idx].append(result["eval_micro_accuracy"])
        all_results[idx].append(result['eval_classwise_accuracy_yes'])
        all_results[idx].append(result['eval_classwise_accuracy_no'])
        all_results[idx].append(result['eval_classwise_accuracy_more'])

with open("/research/dept7/ik_grp/ablation_discern_like_results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(all_results)













