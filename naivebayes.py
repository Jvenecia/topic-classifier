# naivebayes.py
"""Classify written text based on the words appearing in it using a Naïve Bayes model."""

import argparse
import os
import pdb
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = 'data'  # change to path where data is stored the current directory of this file
THIS = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description="Use a Naïve Bayes model to classify text.")
parser.add_argument('-xtrain', '--training_data',
                    help='path to training data file, defaults to ROOT/training_data.txt',
                    default=os.path.join(ROOT, 'training_data.txt'))
parser.add_argument('-ytrain', '--training_labels',
                    help='path to training labels file, defaults to ROOT/training_labels.txt',
                    default=os.path.join(ROOT, 'training_labels.txt'))
parser.add_argument('-xtest', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testing_data.txt',
                    default=os.path.join(ROOT, 'testing_data.txt'))
parser.add_argument('-ytest', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testing_labels.txt',
                    default=os.path.join(ROOT, 'testing_labels.txt'))
parser.add_argument('-t', '--topics',
                    help='path to file containing possible topics, defaults to ROOT/topics.txt',
                    default=os.path.join(ROOT, 'topics.txt'))
parser.add_argument('-v', '--vocabulary',
                    help='path to vocabulary file, defaults to ROOT/vocabulary.txt',
                    default=os.path.join(ROOT, 'vocabulary.txt'))
parser.add_argument('-s', '--stopwords',
                    help='path to file containing stop words to ignore, defaults to ROOT/stopwords.txt',
                    default=os.path.join(ROOT, 'stopwords.txt'))
parser.add_argument('--save', action='store_true',
                    help='save model results to file')


def main(args):
    print("Training a Naïve Bayes Classifier to Identify Text")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    topics_path = os.path.expanduser(args.topics)
    vocabulary_path = os.path.expanduser(args.vocabulary)
    stopwords_path = os.path.expanduser(args.stopwords)

    # Load data from relevant files
    print(f"Loading training data from: {os.path.basename(training_data_path)}")
    xtrain = np.loadtxt(training_data_path,dtype=str,delimiter='\t')

    print(f"Loading training labels from: {os.path.basename(training_labels_path)}")
    ytrain = np.loadtxt(training_labels_path, dtype=int)

    print(f"Loading testing data from: {os.path.basename(testing_data_path)}")
    xtest = np.loadtxt(testing_data_path, dtype=str, delimiter='\t')

    print(f"Loading testing labels from: {os.path.basename(testing_labels_path)}")
    ytest = np.loadtxt(testing_labels_path, dtype=int)

    print(f"Loading topics from: {os.path.basename(topics_path)}")
    topics = np.loadtxt(topics_path, dtype=str)

    print(f"Loading vocabulary from: {os.path.basename(vocabulary_path)}")
    vocabulary = np.loadtxt(vocabulary_path, dtype=str)

    print(f"Loading stop words from: {os.path.basename(stopwords_path)}")
    stopwords = np.loadtxt(stopwords_path, dtype=str)

    # Extract useful parameters
    num_training_sentences = len(ytrain)
    num_testing_sentences = len(ytest)
    num_words = len(vocabulary)
    num_topics = len(topics)

    # Get the words in the datasets
    xtrain_words = [extract_words(sentence, stopwords) for sentence in xtrain]
    xtest_words = [extract_words(sentence, stopwords) for sentence in xtest]


    # Pair the data
    train_data = list(zip(xtrain_words,ytrain))

    print("\n=======================")
    print("TRAINING")
    print("=======================")

    # Estimate the prior probabilities
    print("Estimating prior probabilities via MLE...")
    priors = np.array([np.count_nonzero(ytrain==label)/num_training_sentences for label in range(num_topics)])

    if args.save:
        filename = os.path.expanduser(os.path.join(THIS, 'priors.txt'))
        print(f"  Saving to file: {filename}")
        np.savetxt(filename, priors, fmt="%0.5f")

    # Estimate the class conditional probabilities
    print("Estimating class conditional probabilities via MAP...")

    # Make array of size num_words by num_topics
    class_conditionals = np.zeros((num_words,num_topics))


    # Fixed slow loop
    for sentence, label in train_data:
        for word in range(len(sentence)):
            index = np.where(vocabulary == sentence[word])
            class_conditionals[index, label] += 1   

    beta = 1/num_words

    class_conditionals += beta

    sums = np.sum(class_conditionals, axis=0)
    class_conditionals/=sums

    if args.save:
        filename = os.path.expanduser(os.path.join(THIS, 'class_conditionals.txt'))
        print(f"  Saving to file: {filename}")
        np.savetxt(filename, class_conditionals, fmt="%0.10f", delimiter=',')

    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    log_priors = np.log(priors)
    log_class_conditionals = np.log(class_conditionals)


    print("Counting words in each sentence...")
    # Count the words in the test data  

    counts = np.zeros((num_testing_sentences, num_words))
    for sentence in range(num_testing_sentences):
        for word in range(len(xtest_words[sentence])):
            index = np.where(vocabulary == xtest_words[sentence][word])
            counts[sentence, index]+=1

    print("Computing posterior probabilities...")

    # Initialize an array of size num_testing_sentences and initialize each value to the log of the priors
    log_posteriors = [np.copy(log_priors) for i in range(num_testing_sentences)]

    # For every topic
    for i in range(num_topics):
        
        # For every sentence
        for j in range(num_testing_sentences):

            # For every word in the current sentence
            for word in xtest_words[j]:

                idx = np.where(vocabulary==word)

                # If counts is not 0 at the current index
                if counts[j,idx]:

                    # Add the probability to the log posteriors
                    log_posteriors[j][i] += counts[j,idx]*log_class_conditionals[idx,i]

    print("Assigning predictions via argmax...")
    
    # Generate num_testing_sentences predictions
    pred = []
    for sentence in log_posteriors:
        pred.append(np.argmax(sentence))

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics
    accuracy = 0
    for i in range(num_testing_sentences):
        if pred[i] == ytest[i]:
            accuracy+=1    

    accuracy/=num_testing_sentences
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(ytest,pred,labels=range(10))
    print("Confusion matrix:")
    print(cm)

    # pdb.set_trace()  # uncomment for debugging, if needed


def extract_words(sentence:str, stopwords):
    """Convert a string to a numpy array of words. Follow these steps:
    1. Strip any leading/trailing whitespace.
    2. Make everything lowercase.
    3. Remove special characters (except for hyphens, see rule below).
    4. Replace hyphens with a space so that hyphenated words become two separate words.
    5. Split the string on spaces.
    6. Remove stop words.
    7. Remove single-letter 'words', i.e. anything with len(word) = 1.
    8. Remove pure numbers (e.g. 100, 2024, ...) using isnumeric function.

    Return the array of words.
    """

    special_characters = set(['.', ',', '#', '"', "'", ';', '(', ')', '&', '?', '!'])
    sw = set(stopwords)

    # Perform 1,2,4
    sentence = sentence.strip().lower().replace("-", " ")

    # Perform 3
    for character in sentence:
        if character in special_characters:
            sentence = sentence.replace(character, '')

    # Perform 5
    sentence = sentence.split(" ")

    # Perform 6, 7, 8
    end = []
    for word in sentence:
        if not(word in sw or len(word) == 1 or word.isnumeric()):
            end.append(word)

    return end

if __name__ == '__main__':
    main(parser.parse_args())
