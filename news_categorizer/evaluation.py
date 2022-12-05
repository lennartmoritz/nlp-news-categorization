import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics import Recall, Precision, F1Score


def evaluate(all_labels, all_predictions, run_name):
    labels_unique = (torch.unique(all_labels)).tolist()
    labels_number = len(labels_unique)
    print('Number of labels/classes: %d - which are %s' % (labels_number, labels_unique))
    labels_occurences = Counter(all_labels.tolist())
    print('Occurences of labels/classes: %s' % (labels_occurences))

    ## Specificity
    resulting_confusion_matrix = confusion_matrix(all_labels, all_predictions)
    FP = resulting_confusion_matrix.sum(axis=0) - np.diag(resulting_confusion_matrix)
    FN = resulting_confusion_matrix.sum(axis=1) - np.diag(resulting_confusion_matrix)
    TP = np.diag(resulting_confusion_matrix)
    TN = resulting_confusion_matrix.sum() - (FP + FN + TP)

    # === All types of evaluations ===

    # Evaluation on scope of class level
    TPR = TP / (TP + FN)  # Sensitivity, hit rate, recall, or true positive rate
    TNR = TN / (TN + FP)  # Specificity or true negative rate
    PPV = TP / (TP + FP)  # Precision or positive predictive value
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # Fall out or false positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # Overall accuracy
    F1S = 2 * (PPV * TPR) / (PPV + TPR)  # F1-Score
    evaluations = [FP, FN, TP, TN, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, F1S]

    # Evaluation on scope above of class level
    FPa = sum(FP)  # Summing the values for all classes together
    FNa = sum(FN)
    TPa = sum(TP)
    TNa = sum(TN)
    TPRa = TPa / (TPa + FNa)  # Sensitivity, hit rate, recall, or true positive rate
    TNRa = TNa / (TNa + FPa)  # Specificity or true negative rate
    PPVa = TPa / (TPa + FPa)  # Precision or positive predictive value
    NPVa = TNa / (TNa + FNa)  # Negative predictive value
    FPRa = FPa / (FPa + TNa)  # Fall out or false positive rate
    FNRa = FNa / (TPa + FNa)  # False negative rate
    FDRa = FPa / (TPa + FPa)  # False discovery rate
    ACCa = (TPa + TNa) / (TPa + FPa + FNa + TNa)  # Overall accuracy
    F1Sa = 2 * (PPVa * TPRa) / (PPVa + TPRa)  # F1-Score
    evaluationsa = [FPa, FNa, TPa, TNa, TPRa, TNRa, PPVa, NPVa, FPRa, FNRa, FDRa, ACCa, F1Sa]

    # Macro recall, precision, f1-score
    rec = Recall('multiclass', num_classes=4, average='macro')
    macro_recall = rec(all_predictions, all_labels)
    prec = Precision('multiclass', num_classes=4, average='macro')
    macro_precision = prec(all_predictions, all_labels)
    f1 = F1Score('multiclass', num_classes=4, average='macro')
    macro_f1 = f1(all_predictions, all_labels)
    macro_evaluations = [macro_recall, macro_precision, macro_f1]

    # Save evaluations to file
    save_eval = False

    if save_eval:
        os.makedirs("./evaluations", exist_ok=True)
        filename = 'evaluations-' + run_name + '.txt'
        current_file = os.path.join("evaluations", filename)
        with open(current_file, 'w') as file:
            file.write(str(labels_occurences) + "\n")
            file.write("\n")
            for element in evaluations:
                file.write(str(element) + "\n")
            file.write("\n")
            for element in evaluationsa:
                file.write(str(element) + "\n")
            file.write("\n")
            for element in macro_evaluations:
                file.write(str(element) + "\n")

    # Display evaluations on console
    display_eval = True

    if display_eval:
        print('Macro-average:')
        print(f"{'Recall:':<55}{macro_recall}")
        print(f"{'Precision:':<55}{macro_precision}")
        print(f"{'F1-Score:':<55}{macro_f1}")

