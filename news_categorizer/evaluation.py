import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def evaluate(all_labels, all_predictions, list_of_classes, embedding, train_size, num_epochs):
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

    # Save evaluations to file
    save_eval = False

    if save_eval:
        os.makedirs("./evaluations", exist_ok=True)
        current_classes_as_string = ""
        for class_key in list_of_classes:
            current_classes_as_string = current_classes_as_string + class_key
        current_filename = "evaluation-" + str(embedding) + "-" + current_classes_as_string + "-" + str(train_size) + "-" + str(num_epochs)
        current_file = "./evaluations/" + current_filename + ".txt"
        with open(current_file, 'w') as file:
            file.write(str(labels_occurences) + "\n")
            file.write("\n")
            for element in evaluations:
                file.write(str(element) + "\n")
            file.write("\n")
            for element in evaluationsa:
                file.write(str(element) + "\n")

    # Display evaluations on console
    display_eval = True

    if display_eval:
        # Evaluation on scope of class level
        print("Evaluation on scope of class level")
        print(f"FP: {FP}     FN: {FN}     TP: {TP}     TN: {TN}")

        print(f"{'Sensitivity, hit rate, recall, or true positive rate:':<55}{TPR}")
        print(f"{'Specificity or true negative rate:':<55}{TNR}")
        print(f"{'Precision or positive predictive value:':<55}{PPV}")
        print(f"{'Negative predictive value:':<55}{NPV}")
        print(f"{'Fall out or false positive rate:':<55}{FPR}")
        print(f"{'False negative rate:':<55}{FNR}")
        print(f"{'False discovery rate:':<55}{FDR}")
        print(f"{'Overall accuracy:':<55}{ACC}")
        print(f"{'F1-Score:':<55}{F1S}")

        print("")

        # Evaluation on scope above of class level
        print("Evaluation on scope above of class level")
        print(f"FP: {FPa}     FN: {FNa}     TP: {TPa}     TN: {TNa}")

        print(f"{'Sensitivity, hit rate, recall, or true positive rate:':<55}{TPRa:>12}")
        print(f"{'Specificity or true negative rate:':<55}{TNRa:>12}")
        print(f"{'Precision or positive predictive value:':<55}{PPVa:>12}")
        print(f"{'Negative predictive value:':<55}{NPVa:>12}")
        print(f"{'Fall out or false positive rate:':<55}{FPRa:>12}")
        print(f"{'False negative rate:':<55}{FNRa:>12}")
        print(f"{'False discovery rate:':<55}{FDRa:>12}")
        print(f"{'Overall accuracy:':<55}{ACCa:>12}")
        print(f"{'F1-Score:':<55}{F1Sa:>12}")
