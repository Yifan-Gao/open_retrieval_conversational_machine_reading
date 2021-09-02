import json
import os

if os.path.exists("/research/dept7/ik_grp/all_results.json"):
    with open("/research/dept7/ik_grp/all_results.json") as f:
        all_results = json.load(f)
else:
    all_results = {}

# for ltrans in ["0","1","2","3","4","5"]:
#     for lentail in ["0.0", "1.0", "2.0", "4.0", "6.0", "8.0", "10.0",]:
#         for seed in ["27", "95", "19", "87", "11"]:
#             path = f"./out/v9_roberta-base_Lentail{lentail}_Ltrans{ltrans}_seed{seed}"
#             for split in ['dev', 'test']:
#                 if "-".join([ltrans, lentail, seed, split]) in all_results:
#                     continue
#                 try:
#                     with open(path+f"/results_{split}.json") as f:
#                         result = json.load(f)
#                     all_results["-".join([ltrans, lentail, seed, split])] = {
#                         "eval_macro_accuracy": result["eval_macro_accuracy"],
#                         "eval_micro_accuracy": result["eval_micro_accuracy"],
#                         "eval_classwise_accuracy_more": result["eval_classwise_accuracy_more"],
#                         "eval_classwise_accuracy_no": result["eval_classwise_accuracy_no"],
#                         "eval_classwise_accuracy_yes": result["eval_classwise_accuracy_yes"],
#                     }
#                 except:
#                     print(path+f"results_{split}.json")
#
# with open("/research/dept7/ik_grp/all_results.json", 'w') as f:
#     json.dump(all_results, f)


# import csv
# best_results = [[] for _ in range(10)]
# lentail="8.0"
# ltrans="4"
#
# for split in ['dev', 'test']:
#     for idx, seed in enumerate(["27", "95", "19", "87", "11"]):
#         if split == 'test': idx += 5
#         key = "-".join([ltrans, lentail, seed, split])
#         best_results[idx].append(all_results[key]["eval_macro_accuracy"])
#         best_results[idx].append(all_results[key]["eval_micro_accuracy"])
#         best_results[idx].append(all_results[key]['eval_classwise_accuracy_yes'])
#         best_results[idx].append(all_results[key]['eval_classwise_accuracy_no'])
#         best_results[idx].append(all_results[key]['eval_classwise_accuracy_more'])
# with open("/research/dept7/ik_grp/best_results.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(best_results)
#
#
#
# import csv
# import numpy as np
# hyp_results = [[] for _ in range(14)]
#
# for split in ['dev', 'test']:
#     for idx, lentail in enumerate(["0.0", "1.0", "2.0", "4.0", "6.0", "8.0", "10.0",]):
#         if split == 'test': idx += 7
#         for ltrans in ["0", "1", "2", "3", "4", "5"]:
#             curr_results = [all_results["-".join([ltrans, lentail, seed, split])]["eval_macro_accuracy"]*all_results["-".join([ltrans, lentail, seed, split])]["eval_micro_accuracy"]/100 for seed in ["27", "95", "19", "87", "11"]]
#             hyp_results[idx].append(float("{:.2f}".format(np.mean(curr_results))))
#
# with open("/research/dept7/ik_grp/all_results.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(hyp_results)


import csv
baseline_results = [[] for _ in range(10)]
lentail="0.0"
ltrans="4"

for split in ['dev', 'test']:
    for idx, seed in enumerate(["27", "95", "19", "87", "11"]):
        if split == 'test': idx += 5
        key = "-".join([ltrans, lentail, seed, split])
        baseline_results[idx].append(all_results[key]["eval_macro_accuracy"])
        baseline_results[idx].append(all_results[key]["eval_micro_accuracy"])
        baseline_results[idx].append(all_results[key]['eval_classwise_accuracy_yes'])
        baseline_results[idx].append(all_results[key]['eval_classwise_accuracy_no'])
        baseline_results[idx].append(all_results[key]['eval_classwise_accuracy_more'])
with open("/research/dept7/ik_grp/baseline_results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(baseline_results)

    











