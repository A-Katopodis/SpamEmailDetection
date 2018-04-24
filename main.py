from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from math import log
import random


# This load the emails file use naive bayes for this
def load_corpora(path):
    emails = []
    y = []
    print ("Loading Dataset.....")
    pus = listdir(path)
    for pu in pus:
        parts = listdir(path+"/"+pu)
        for part in parts:
            openFile = path+"/"+pu+"/"+part
            files = listdir(path+"/"+pu+"/"+part)
            for email in files:
                if "legit" in email:
                    y.append(1)
                else:
                    y.append(0)
                f = open(openFile+"/"+email,'r')
                emails.append(f.read())
                f.close()
    # emails = np.array(emails)
    # y = np.array(y)
    print ("Finished Loading Dataset!")

    return emails, y


# return the key with the max value
def max_key(dict):
    values = list(dict.values())
    keys = list(dict.keys())
    return keys[values.index(max(values))]


def get_probabilities(X_train, y_train):
    N = len(X_train)
    categories = set(y_train)

    # The number of data in each category
    X_in_category = {i: y_train.count(i) for i in categories}

    pr_W_category = {}

    # The probability of each category
    pr_category = {i: X_in_category[i] / N for i in categories}

    # The number of data that belongs to each category

    # First zero, we don't know how many W each category contains
    W_in_category = {i: 0 for i in categories}

    # We calculate the number of times a W is in each category
    # and we add bias for laplace
    for email, category in zip(X_train, y_train):
        # removes empty space's from the train data
        # and store them in an array
        words = email.split()
        for word in words:

            W_in_category[category] += 1
            # If we don't have the word in the dict we add it
            if word not in pr_W_category:
                # We add the bias +1
                pr_W_category[word] = {i: 1 for i in categories}

            # we increase the count per category
            pr_W_category[word][category] += 1

    # the total number of distinct w across all data
    V = len(pr_W_category)

    # we compute the probabilities for each distinct W
    for W in pr_W_category:
        for category in pr_W_category[W]:
            pr_W_category[W][category] /= W_in_category[category] + V

    #------------------------------------------------#
    # We have computed the  probability for category ( pr_category )
    # and the likehood of each w for each category (pr_W_category)
    return pr_category, pr_W_category


def predict_category(data, pr_W_category, pr_category):
    elements = data.split()
    new_pr_category = {i: 0 for i in pr_category}
    # we compute the probability for each category
    for category in pr_category:
        for w in elements:
            if w in pr_W_category:
                # We use log because it keeps the range between [0,1]
                new_pr_category[category] += log(pr_W_category[w][category])
        new_pr_category[category] += log(pr_category[category])

    return max_key(new_pr_category)


def compute_probabilities(X_train, y_train):
    N = len(X_train)
    spam_mails = y_train.count(-1)
    legit_mails = y_train.count(1)
    num_words_legit = 0
    num_words_spam = 0
    # The features represent the category of each word
    features = {}

    for i in range(N):
        # terms= word
        words = X_train[i].split()
        if y_train[i] == 1:  # If the label is 0 it is a ham
            for word in words:
                num_words_legit += 1
                if word in features:
                    features[word]['ham'] += 1
                else:
                    features[word] = {'ham': 1, 'spam': 0}

        else:  # If the label is 1 it is a spam
            for word in words:
                num_words_spam += 1
                if word in features:
                    features[word]['spam'] += 1
                else:
                    features[word] = {'ham': 0, 'spam': 1}
    print ("first: ",num_words_legit, num_words_spam)
    for term in features:
        features[term]['ham'] += 1
        features[term]['spam'] += 1
        features[term]['ham'] /= float(num_words_legit + len(features))
        features[term]['spam'] /= float(num_words_spam + len(features))

    pr_ham = legit_mails / N
    pr_spam = spam_mails / N

    return features, pr_ham, pr_spam


def split_data(x, y, k=10):
    convert = False
    if type(x) == type(np.asarray([])):
        x = x.tolist()
        y = y.tolist()
        convert = True


    c = list(zip(x, y))

    # shuffle the array so we wont get the same
    # training and test data each time
    random.shuffle(c)


    # unpack the values
    a, b = zip(*c)

    N = len(x)
    pr = 1/k
    # the percentage of the train data
    perc_of_train = 1- int(pr*N)
    # Since we shuffle it the first elements will never be in the same order twice
    X_train = a[:perc_of_train]
    y_train = b[:perc_of_train]

    X_test = a[perc_of_train:]
    y_test = b[perc_of_train:]

    if convert:
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

    return X_train, y_train , X_test, y_test


def naiveBayes(X,y):
    k = 10

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(k):
        X_train, y_train, X_test, y_test = split_data(X, y, k)
        # train the model
        pr_category, pr_W_category = get_probabilities(X_train, y_train)

        predictions = []
        for data in X_test:
            predictions.append(predict_category(data, pr_W_category, pr_category))
        precision, recall, accuracy = calculate_scores(y_test, predictions)
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)

    # ta score pou 8elei
    average_recall = np.mean(recall_scores)
    average_precision = np.mean(precision_scores)
    average_accuracy = np.mean(accuracy_scores)
    print (average_accuracy, average_precision, average_recall)

    # edv apla svste to plot an 8elete kai ta print
    plt.plot(np.squeeze(accuracy_scores))
    plt.title("Accuracy scores")
    plt.show()


def run_naiveBayes(dataset = "pu_corpora_public"):
    pu_corpora = dataset
    # return X_train and y_train, 1 is legit, 0 is spam
    X, y = load_corpora(pu_corpora)
    # print "X train shape: ", X_train.shape, "y_train: ", y_train.shape

    print ("Total email: ", len(X))

    naiveBayes(X, y)


def calc_categories_scores(y_train, predictions):
    # it assumes a numpy array as input
    categories_scores = []
    # for each category
    for i in range(y_train.shape[1]):
        categories_scores.append(calculate_scores(y_train[:,i], predictions[:,i]))
    return categories_scores


def calculate_scores(y_train, predictions):
    # true positive, true negative, false positive, false negative
    tp, tn,fp, fn = 0, 0, 0,0
    # we assume we have only 1 and 0
    for true, prediction in zip(y_train,predictions):
        if true == 1 and prediction == 1:
            tp += 1
        elif true == 1 and prediction == 0:
            fn += 1
        elif true == 0 and prediction == 1:
            fp += 1
        else:
            tn += 1
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, accuracy



# run_stohastic_logistic_regression()
# plt.show()
run_naiveBayes()

