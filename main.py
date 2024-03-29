import numpy as np

from algorithms.preprocessing import preprocess
from algorithms.riseAlgorithm import riseAlgorithm


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    # HYPERPARAMETERS **********************************
    dta_option1 = 1  # Number of the Dataset
    perc_test = 0.3  # Fraction of Test set
    # **************************************************

    print('\033[1m' + 'The number of the dataset selected is: ' + str(dta_option1) + '\033[0m')
    print('********************************************** Rules induced with Train set')
    # Preprocess the data
    rows, numericalatt, labels = preprocess(dta_option1)
    data = np.array(rows)
    data = data.astype(np.float)
    labels = (np.array(labels)).reshape(data.shape[0])
    labels = labels.astype(np.int)

    # Shuffle data, and divide train and test set
    indicc = np.arange(data.shape[0])
    np.random.shuffle(indicc)
    cut = round(perc_test * data.shape[0])
    data = np.array(data)
    data_test = data[indicc[0:cut], :]
    labels_test = labels[indicc[0:cut]]
    data_trn = data[indicc[cut:], :]
    labels_trn = labels[indicc[cut:]]

    # Call fit and predict of the algorithm (RISE) I have implemented
    rise = riseAlgorithm(numericalatt)
    rise.fit(data_trn, labels_trn)
    rise.classify(data_test)

    # Compute Accuracy between ground truth and predicted labels
    test_accuracy = (sum([a == b for a,b in zip(labels_test, rise.tst_labels)]))/len(labels_test)
    print('\033[1m' + 'The final test accuracy is: ' + str(round(test_accuracy, 3)) + '\033[0m')

    # Find rules of the whole model using train and test set
    print('\n********************************************** Rules induced with the full set (Train + Test)')
    rise_final = riseAlgorithm(numericalatt)
    rise_final.fit(data, labels)

    # Print rules in screen with coverage and precision
    bin = int(input('\nDo you want to see the rules? (0-No, 1-Yes). Introduce the number: '))
    if bin == 1:
        rise_final.print_rules()


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()