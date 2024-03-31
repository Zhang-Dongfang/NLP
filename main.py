import argparse
import math
import pickle
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")


class LogLinearModel:
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.zeros((n_classes, n_features))

    def train(self, X, y, lr=0.01, epochs=20, batch_size=64):
        n_samples = X.shape[0]
        for epoch in tqdm(range(epochs)):
            batch_losses = []
            shuffled_indices = np.random.permutation(n_samples)
            for start_index in tqdm(range(0, n_samples, batch_size), leave=False):
                end_index = min(start_index + batch_size, n_samples)
                batch_indices = shuffled_indices[start_index:end_index]

                batch_X = X[batch_indices].toarray()
                batch_y = y[batch_indices]

                scores = batch_X.dot(self.weights.T)
                probs = self._softmax(scores)

                loss = self._cross_entropy(probs, batch_y)
                batch_losses.append(loss)

                delta = (probs - batch_y).T.dot(batch_X)
                self.weights -= lr * delta

            epoch_loss = np.mean(batch_losses)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, X):
        scores = X.dot(self.weights.T)
        probs = self._softmax(scores)
        return np.argmax(probs, axis=1)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def _cross_entropy(self, probs, y_true):
        log_probs = -np.log(probs[range(len(probs)), np.argmax(y_true, axis=1)])
        return np.mean(log_probs)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


def replace_space(word):
    return re.sub(r"[-\\/&]", " ", word)


def replace_num(word):
    return re.sub(r"\d+", "<NUM>", word)


def separate_num(word):
    return re.sub(r"(<NUM>)", r" \1 ", word)


def tokenize(text):
    return text.split()


def lower(tokens):
    return [word.lower() for word in tokens]


def remove_word_suffixes(word):
    if word.endswith("'s"):
        word = word[:-2]
    else:
        return re.sub(r'[.,:()\'"?;#$!]', "", word)


def remove_suffixes(tokens):
    return [remove_word_suffixes(word) for word in tokens]


def remove_stopwords(tokens, stopwords):
    return [word for word in tokens if (word not in stopwords) and (word is not None)]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_with_pos(tokens):
    pos_tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tagged
    ]


def compute_tf(tokens):
    tf = Counter(tokens)
    for i in tf:
        tf[i] = (1 + math.log10(tf[i])) if tf[i] != 0 else 0
    return dict(tf)


def compute_idf(dft, df_tokens_len):
    return math.log10(df_tokens_len / dft)


def f1_score(y_true, y_pred):
    precisions = []
    recalls = []
    for label in np.unique(y_true):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    f1_scores = []
    for precision, recall in zip(precisions, recalls):
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        f1_scores.append(f1)

    weights = np.bincount(y_true) / len(y_true)
    weighted_f1 = np.sum(f1_scores * weights)

    return weighted_f1


def preprocess_data(df, stopwords):
    df["title"] = df["title"].apply(replace_space)
    df["description"] = df["description"].apply(replace_space)
    df["title"] = df["title"].apply(replace_num)
    df["description"] = df["description"].apply(replace_num)
    df["title"] = df["title"].apply(separate_num)
    df["description"] = df["description"].apply(separate_num)
    df["tokens"] = df["title"].apply(tokenize) + df["description"].apply(tokenize)
    df.drop("description", axis=1, inplace=True)
    df.drop("title", axis=1, inplace=True)
    df["tokens"] = df["tokens"].apply(lower)
    df["tokens"] = df["tokens"].apply(remove_suffixes)
    df["tokens"] = df["tokens"].apply(lambda x: remove_stopwords(x, stopwords))
    df["tokens"] = df["tokens"].apply(lemmatize_with_pos)
    return df


def compute_tf_idf(df, idf=None):
    TF = [compute_tf(tokens) for tokens in df["tokens"]]
    if idf is None:
        words_counter = Counter()
        for tokens in df["tokens"]:
            words_counter.update(tokens)
        vocabulary = dict(words_counter)
        idf = {
            word: compute_idf(dft, len(df["tokens"]))
            for word, dft in vocabulary.items()
        }
    word_list = list(idf.keys())
    word_to_index = {word: i for i, word in enumerate(word_list)}
    data = []
    indices = []
    indptr = [0]
    for i in range(len(TF)):
        for word, tf in TF[i].items():
            if word in idf:
                tf_idf = tf * idf[word]
                data.append(tf_idf)
                indices.append(word_to_index[word])
        indptr.append(len(data))
    tf_idf_matrix = csr_matrix(
        (data, indices, indptr), shape=(len(TF), len(idf)), dtype=float
    )
    return tf_idf_matrix, idf


def main(args):
    if args.mode == "train":
        df = pd.read_csv(
            args.train_data, header=None, names=["label", "title", "description"]
        )
    else:
        df = pd.read_csv(
            args.test_data, header=None, names=["label", "title", "description"]
        )
    with open(args.stopwords) as file:
        stopwords = file.read().split(",")

    df = preprocess_data(df, stopwords)
    tf_idf = compute_tf_idf(df)
    y = np.eye(4)[df["label"] - 1]

    if args.mode == "train":
        tf_idf, idf = compute_tf_idf(df)
        with open(args.idf_path, "wb") as f:
            pickle.dump(idf, f)
    else:
        with open(args.idf_path, "rb") as f:
            idf = pickle.load(f)
        tf_idf, _ = compute_tf_idf(df, idf)

    y = np.eye(4)[df["label"] - 1]

    if args.mode == "train":
        X_train, X_val, y_train, y_val = train_test_split(
            tf_idf, y, test_size=0.2, random_state=42
        )
        model = LogLinearModel(X_train.shape[1], 4)
        model.train(X_train, y_train, epochs=args.epochs)
        model.save(args.model_path)
        val_predictions = model.predict(X_val)
        val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        f1 = f1_score(np.argmax(y_val, axis=1), val_predictions)
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(np.argmax(y_val, axis=1), val_predictions))
    else:
        loaded_model = LogLinearModel.load(args.model_path)
        val_predictions = loaded_model.predict(tf_idf)
        val_accuracy = np.mean(val_predictions == np.argmax(y, axis=1))
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        f1 = f1_score(np.argmax(y, axis=1), val_predictions)
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(np.argmax(y, axis=1), val_predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear-Log Model for Text Classification"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Running mode: train or test",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="ag_news_csv/train.csv",
        help="Path to the training data",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="ag_news_csv/test.csv",
        help="Path to the test data",
    )
    parser.add_argument(
        "--stopwords",
        type=str,
        default="stopwords.txt",
        help="Path to the stopwords file",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pkl",
        help="Path to save/load the model",
    )
    parser.add_argument(
        "--idf_path",
        type=str,
        default="IDF.pkl",
        help="Path to save/load the IDF dictionary",
    )
    args = parser.parse_args()
    main(args)
