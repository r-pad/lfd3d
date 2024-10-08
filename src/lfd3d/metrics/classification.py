import pandas as pd
import torchmetrics.functional.classification as tfc


def get_metrics(preds, labels):
    # "True" accuracy, aka on the true distribution (without considering class imbalance).
    global_acc = tfc.multiclass_accuracy(preds, labels, num_classes=10, average="micro")

    # Per-class accuracy, averaged over all classes with equal weight.
    macro_acc = tfc.multiclass_accuracy(preds, labels, num_classes=10, average="macro")

    # Per-class accuracy, not averaged over all classes.
    class_acc = tfc.multiclass_accuracy(preds, labels, num_classes=10, average="none")

    # Create a dataframe with the per-label accuracies.
    acc_df = pd.DataFrame(class_acc[None], columns=[str(i) for i in range(10)])

    return {
        "global_acc": global_acc,
        "macro_acc": macro_acc,
        "acc_df": acc_df,
    }
