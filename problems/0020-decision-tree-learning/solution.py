import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    lable_bucket = {}
    n = len(labels)
    for lbl in labels:
        lable_bucket[lbl] = lable_bucket.get(lbl, 0) + 1
    
    entropy = 0
    for key, value in lable_bucket.items():
        p = lable_bucket[key]/n
        entropy +=  (-1*p*math.log2(p))
    
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    lables = []
    for exmpl in examples:
        lables.append(exmpl[target_attr])

    final_entropy = calculate_entropy(lables)

    child_lables = {}
    for exmpl in examples:
        child_lables[exmpl[attr]] = child_lables.get(exmpl[attr], [])
        child_lables[exmpl[attr]].append(exmpl[target_attr])

    for key, key_lables in child_lables.items():
        final_entropy = final_entropy - ((len(key_lables)/len(lables))*calculate_entropy(key_lables))
    
    return final_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:

    attributes = []
    for key, value in examples[0].items():
        if key != target_attr:
            attributes.append(key)

    attributes.sort()

    information_gain = []
    for attr in attributes:
        information_gain.append(calculate_information_gain(examples, attr, target_attr))

    max_index = 0
    max_ig = information_gain[0]
    for i in range(len(information_gain)):
        if information_gain[i] > max_ig:
            max_ig = information_gain[i]
            max_index = i

    return attributes[max_index]

    

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""

    if len(attributes) == 0:
        return 'No'

    lbl = set()
    for exmpl in examples:
        lbl.add(exmpl[target_attr])
    
    if len(lbl) <= 1:
        return {}

    key = majority_class(examples, target_attr)
    values = {}
    for exmpl in examples:
        new_exmpl = {k: v for k, v in exmpl.items() if k != key}
        values[exmpl[key]] = values.get(exmpl[key], [])
        values[exmpl[key]].append(new_exmpl)
    
    values = dict(sorted(values.items()))
    tree = {}
    tree[key] = {}
    for value, new_exmpls in values.items():
        child_tree = learn_decision_tree(new_exmpls, attributes, target_attr)
        if child_tree:
            tree[key][value] = child_tree
        else:
            tree[key][value] = new_exmpls[0][target_attr]
    
    return dict(sorted(tree.items()))