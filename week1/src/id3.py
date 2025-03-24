from src.utils import best_split, split_dataset, most_common_label


class Node:
    def __init__(self, column=None, value=None, true_branch=None, false_branch=None, label=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.label = label

    def is_leaf(self):
        return self.true_branch is None and self.false_branch is None


def build_tree(data, target):
    if len(data[target].unique()) == 1:
        return Node(label=data[target].iloc[0])

    # Dacă nu mai sunt atribute disponibile, returnăm cel mai frecvent label
    if len(data.drop(columns=[target]).columns) == 0:
        return Node(label=most_common_label(data[target]))

    # Alegem cel mai bun atribut pentru împărțire
    best_column, best_value = best_split(data, target)

    if best_column is None:
        return Node(label=most_common_label(data[target]))

    true_branch_data, false_branch_data = split_dataset(data, best_column, best_value)
    true_branch = build_tree(true_branch_data, target)
    false_branch = build_tree(false_branch_data, target)

    return Node(column=best_column, value=best_value, true_branch=true_branch, false_branch=false_branch)
