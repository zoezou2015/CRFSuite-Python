
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite



def load_data(filename):
    file = open(filename)
    i = 0
    listOfSen = []
    listOfChunk = []
    for line in file:
        i = i+1
        if line != "\n":
            lines = line.strip("\n").split(" ")
            if len(lines) == 3:
                entry = (lines[0], lines[1], lines[2])
                listOfChunk.append(entry)
        elif line == "\n":
            # print "null"
            listOfSen.append(listOfChunk)
            listOfChunk = []
        #
        # if i>50:
        #     break

    file.close()
    return listOfSen


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        # 'word.lower=' + word.lower(),
        # 'word[-3:]=' + word[-3:],
        # 'word[-2:]=' + word[-2:],
        # 'word.isupper=%s' % word.isupper(),
        # 'word.istitle=%s' % word.istitle(),
        # 'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        # 'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            # '-1:word.lower=' + word1.lower(),
            # '-1:word.istitle=%s' % word1.istitle(),
            # '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            # '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            # '+1:word.lower=' + word1.lower(),
            # '+1:word.istitle=%s' % word1.istitle(),
            # '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            # '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

def train(X_train, y_train,model):
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(model)
    print trainer.get_params()
    return model

def predict(X_test, model_name):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_name)
    Y_pred = [tagger.tag(xseq) for xseq in X_test]
    return Y_pred

def evaluation(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )

if __name__ == "__main__":
    train_data = 'data/train.data.txt'
    test_data = 'data/test.data.txt'
    model_path = 'model/conll2000-en.pycrfsuite'
    train_sents = load_data(train_data)
    test_sents = load_data(test_data)

    # train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    # test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    model = train(X_train, y_train, model_path)
    y_pred = predict(X_test, model)
    print(evaluation(y_test, y_pred))




    print sent2features(train_sents[0])[0]
    # print train_sents[0]

