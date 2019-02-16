from collections import Counter
import os

def make_dictionary():
    direc = './email/'
    files = os.listdir(direc)

    emails = [direc + email for email in files]
    words = []
    c = len(emails)

    for email in emails:
        f = open(email, encoding='ISO 8859-1')
        blob = f.read()
        words += blob.split(' ')
        print(c)
        c -= 1
    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ''

    dictionary = Counter(words)
    del dictionary['']
    return dictionary.most_common(3000)

def make_dataset(dictionary):
    direct = './email/'
    files = os.listdir(direct)
    emails = [direct + email for email in files]
    feature_set = []
    labels = []
    c = len(emails)
    for email in emails:
        print(email)
        data = []
        f = open(email, encoding='ISO 8859-1')
        words = f.read().split(' ')
        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)

        if 'ham' in email:
            labels.append(0)
        if 'spam' in email:
            labels.append(1)
        print(c)
        c -= 1
    return feature_set, labels



dictionary = make_dictionary()
features, labels = make_dataset(dictionary)