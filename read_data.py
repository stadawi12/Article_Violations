from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset

plt.rcParams.update({'font.size': 10})

dataset = load_dataset("lex_glue", "ecthr_a")

# all train, test and valid cases with labels
train_set: list = dataset["train"]
test_set: list = dataset["test"]
valid_set: list = dataset["validation"]

def get_case_info(dataset: list, case: int) -> Tuple[int, list]:
    # dictionary containing a single case and its labels
    case_example: dict = dataset[case]

    # a list of facts conjugate to a case
    case_text: list = case_example["text"]
    # label conjugate to a case
    case_label: list = case_example["labels"]

    return (len(case_text), case_label)

def get_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        label = dataset[i]["labels"]
        labels.append(label)
    return labels

def flatten(l):
    return [item for sublist in l for item in sublist]

train_labels = get_labels(train_set)
test_labels = get_labels(test_set)
valid_labels = get_labels(valid_set)

def count_articles(article, labels):
    count = 0
    for subset in labels:
        if article in subset:
            count += 1
    return count

def count_empty(labels):
    count = 0
    for subset in labels:
        if len(subset) == 0:
            count += 1
    return count

counts = []
all_labels = [train_labels, test_labels, valid_labels]
for dataset in all_labels:
    count = []
    for i in range(10):
        count.append(count_articles(i, dataset) / len(dataset)) 
    count.append(count_empty(dataset) / len(dataset))
    counts.append(count)

train_counts = counts[0]
test_counts = counts[1]
valid_counts = counts[2]

x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'none']
x_axis = np.arange(11)

fig, ax = plt.subplots(1, 1, figsize = (5, 3), dpi=80)
ax.bar(x_axis - 0.2, train_counts, 0.2, label='train')
ax.bar(x_axis, test_counts, 0.2, label='test')
ax.bar(x_axis + 0.2, valid_counts, 0.2, label='valid')
plt.xlabel("Article")
plt.xticks(x_axis, x)
plt.ylabel("Proportion of Counts (%)")
plt.title("Label proportion in each dataset")
plt.legend()
plt.savefig("docs/label_proportions.pdf", format="pdf", 
        bbox_inches='tight', dpi=80)
plt.show()

# def relational_count(article1, article2, labels):
    # count = 0
    # for subset in labels:
        # if article1 in subset and article2 in subset:
            # count += 1
    # return count

# train_count = relational_count(0, 1, train_labels)
# print(train_count)


# train_unique = set(flatten(train_labels))
# test_unique = set(flatten(test_labels))
# valid_unique = set(flatten(valid_labels))

# all_labels = [train_labels, test_labels, valid_labels]
# all_unique = set(flatten(flatten(all_labels)))

# print(train_unique)
# print(test_unique)
# print(valid_unique)
# print(all_unique)

# print(len(flatten(train_labels))/len(train_labels))
# print(len(flatten(test_labels))/len(test_labels))
# print(len(flatten(valid_labels))/len(valid_labels))

# Making box plots
# all_facts = []
# 
# sets = [train_set, test_set, valid_set]
# 
# for s in sets:
#     set_facts = []
#     for case in range(len(s)):
#         no_case_facts, case_label = get_case_info(s, case)
#         if no_case_facts <= 700:
#             set_facts.append(no_case_facts)
#     all_facts.append(set_facts)
# 
# fig, ax = plt.subplots(1, 1, figsize=(3,3), dpi=100)
# ax.boxplot(all_facts)
# plt.xticks([1,2,3], ["train", "test", "valid"])
# plt.ylabel("No. of Facts")
# plt.xlabel("Dataset")
# ax.set_ylim([0,80])
# plt.savefig("docs/facts_boxplots_cutoff.pdf", dpi=100, format="pdf", 
#         bbox_inches='tight')
# plt.show()

# # Table statistics
# all_facts = []
# 
# sets = [train_set, test_set, valid_set]
# 
# for s in sets:
#     set_facts = []
#     for case in range(len(s)):
#         no_case_facts, case_label = get_case_info(s, case)
#         set_facts.append(no_case_facts)
#     all_facts.append(set_facts)
# 
# all_facts = np.array(all_facts)
# names = ["train", "test", "valid"]
# for name, facts in zip(names, all_facts):
#     print(f"min number of facts in {name}:", np.min(facts))
#     print(f"max number of facts in {name}", np.max(facts))
#     print(f"mean number of facts in {name}", np.mean(facts))
#     print(f"median number of facts in {name}", np.median(facts))
