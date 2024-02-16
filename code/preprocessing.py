from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106",
                             "f107", "f108", "f108.2", "f109", "f110", "f111", "f112",
                             "f113", "f114", "f115", "f116"]  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags.add("*")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            @param: file_path: full path of the file to read
            Updates the histories list. counts the appearances of each feature
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                n = len(split_words)
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    # f100
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

                    # f101 , f102
                    prefix_list, suffix_list = get_prefix_suffix_list(cur_word)
                    for curr_pre, curr_suf in zip(prefix_list, suffix_list):
                        if (curr_pre, cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(curr_pre, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(curr_pre, cur_tag)] += 1

                        if (curr_suf, cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(curr_suf, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(curr_suf, cur_tag)] += 1

                    # f103, f104, f105
                    p_word, p_tag = split_words[word_idx - 1].split('_')
                    if word_idx >= 2:
                        _, pp_tag = split_words[word_idx - 2].split('_')
                        if (pp_tag, p_tag, cur_tag) not in self.feature_rep_dict["f103"]:
                            self.feature_rep_dict["f103"][(pp_tag, p_tag, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f103"][(pp_tag, p_tag, cur_tag)] += 1

                    if word_idx >= 1:
                        if (p_tag, cur_tag) not in self.feature_rep_dict["f104"]:
                            self.feature_rep_dict["f104"][(p_tag, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f104"][(p_tag, cur_tag)] += 1

                    if (cur_tag) not in self.feature_rep_dict["f105"]:
                        self.feature_rep_dict["f105"][(cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f105"][(cur_tag)] += 1

                    # f106, f107
                    if word_idx >= 1:
                        if (p_word, cur_tag) not in self.feature_rep_dict["f106"]:
                            self.feature_rep_dict["f106"][(p_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f106"][(p_word, cur_tag)] += 1

                    if word_idx < (len(split_words) - 1):
                        next_word, _ = split_words[word_idx + 1].split('_')
                        if (next_word, cur_tag) not in self.feature_rep_dict["f107"]:
                            self.feature_rep_dict["f107"][(next_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f107"][(next_word, cur_tag)] += 1

                    # f108 - cur/prev word contain 1,2,3, 4, 5, 6, 7+ Capital letters and cur_tag
                    curr_capital_count = (str(min(sum(1 for char in cur_word if char.isupper()), 7)), len(cur_word))
                    p_capital_count = (str(min(sum(1 for char in p_word if char.isupper()), 7)), len(p_word))
                    if (curr_capital_count, cur_tag) not in self.feature_rep_dict["f108"]:
                        self.feature_rep_dict["f108"][(curr_capital_count, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f108"][(curr_capital_count, cur_tag)] += 1
                    if (p_capital_count, cur_tag) not in self.feature_rep_dict["f108.2"]:
                        self.feature_rep_dict["f108.2"][(p_capital_count, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f108.2"][(p_capital_count, cur_tag)] += 1

                    # f109 type of word - number, word, char, mix
                    type_word = str(number_or_word(cur_word))
                    if (type_word, cur_tag) not in self.feature_rep_dict["f109"]:
                        self.feature_rep_dict["f109"][(type_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f109"][(type_word, cur_tag)] += 1

                    # f 110, 111 -  (p_word type, c_tag) and pp_word type, c_tag)
                    if word_idx >= 2:
                        pp_word, pp_tag = split_words[word_idx - 2].split('_')
                        pp_word_type = str(number_or_word(pp_word))
                        if (pp_word_type, cur_tag) not in self.feature_rep_dict["f110"]:
                            self.feature_rep_dict["f110"][(pp_word_type, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f110"][(pp_word_type, cur_tag)] += 1

                    if word_idx >= 1:
                        p_word_type = str(number_or_word(p_word))
                        if (p_word_type, cur_tag) not in self.feature_rep_dict["f111"]:
                            self.feature_rep_dict["f111"][(p_word_type, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f111"][(p_word_type, cur_tag)] += 1

                    # f 112 location of word

                    word_loc = location_in_sen(word_idx, n)
                    if (word_loc, cur_tag) not in self.feature_rep_dict["f112"]:
                        self.feature_rep_dict["f112"][(word_loc, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f112"][(word_loc, cur_tag)] += 1

                    # f113 chars
                    chars = ["-", ",", "$", "%", "!", "?", "...", "--", ":", ";"]
                    if (max(len(cur_word), 12), cur_tag) not in self.feature_rep_dict["f113"]:
                        self.feature_rep_dict["f113"][(max(len(cur_word), 12), cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f113"][(max(len(cur_word), 12), cur_tag)] += 1
                    for char in chars:
                        if char in cur_word:
                            if (char, len(cur_word), cur_tag) not in self.feature_rep_dict["f113"]:
                                self.feature_rep_dict["f113"][(char, len(cur_word), cur_tag)] = 1
                            else:
                                self.feature_rep_dict["f113"][(char, len(cur_word), cur_tag)] += 1

                    # f114 a lot of things
                    if cur_word[0].isupper() and not cur_word[-1].isupper():
                        if (len(cur_word), cur_tag) not in self.feature_rep_dict["f114"]:
                            self.feature_rep_dict["f114"][(len(cur_word), cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f114"][(len(cur_word), cur_tag)] += 1
                        if (cur_tag) not in self.feature_rep_dict["f114"]:
                            self.feature_rep_dict["f114"][(cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f114"][(cur_tag)] += 1

                    # f115, f116
                    if word_idx < (len(split_words) - 1):
                        next_word, _ = split_words[word_idx + 1].split('_')
                        n_word, n_tag = split_words[word_idx + 1].split('_')
                        if (cur_word, cur_tag, n_tag) not in self.feature_rep_dict["f115"]:
                            self.feature_rep_dict["f115"][(cur_word, cur_tag, n_tag)] = 1
                        else:
                            self.feature_rep_dict["f115"][(cur_word, cur_tag, n_tag)] += 1
                        if (p_tag, cur_tag, n_tag) not in self.feature_rep_dict["f115"]:
                            self.feature_rep_dict["f115"][(p_tag, cur_tag, n_tag)] = 1
                        else:
                            self.feature_rep_dict["f115"][(p_tag, cur_tag, n_tag)] += 1
                        if (cur_tag, n_tag) not in self.feature_rep_dict["f115"]:
                            self.feature_rep_dict["f115"][(cur_tag, n_tag)] = 1
                        else:
                            self.feature_rep_dict["f115"][(cur_tag, n_tag)] += 1
                        if (type_word, n_tag) not in self.feature_rep_dict["f115"]:
                            self.feature_rep_dict["f115"][(type_word, n_tag)] = 1
                        else:
                            self.feature_rep_dict["f115"][(type_word, n_tag)] += 1
                        if word_idx >= 2:
                            pp_word, pp_tag = split_words[word_idx - 2].split('_')
                            if (pp_tag, p_tag, cur_tag, n_tag) not in self.feature_rep_dict["f115"]:
                                self.feature_rep_dict["f115"][(pp_tag, p_tag, cur_tag, n_tag)] = 0.5
                            else:
                                self.feature_rep_dict["f115"][(pp_tag, p_tag, cur_tag, n_tag)] += 0.5

                        if word_idx < (len(split_words) - 2):
                            nn_word, nn_tag = split_words[word_idx + 1].split('_')
                            if (cur_tag, n_tag, nn_tag) not in self.feature_rep_dict["f116"]:
                                self.feature_rep_dict["f116"][(cur_tag, n_tag, nn_tag)] = 1
                            else:
                                self.feature_rep_dict["f116"][(cur_tag, n_tag, nn_tag)] += 1
                            if (p_tag, cur_tag, n_tag, nn_tag) not in self.feature_rep_dict["f116"]:
                                self.feature_rep_dict["f116"][(p_tag, cur_tag, n_tag, nn_tag)] = 0.5
                            else:
                                self.feature_rep_dict["f116"][(p_tag, cur_tag, n_tag, nn_tag)] += 0.5
                            if (type_word, n_tag, nn_tag) not in self.feature_rep_dict["f116"]:
                                self.feature_rep_dict["f116"][(type_word, n_tag, nn_tag)] = 1
                            else:
                                self.feature_rep_dict["f116"][(type_word, n_tag, nn_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))
                for i in range(2, len(sentence) - 1):
                    if i + 2 > len(sentence) - 1:
                        nn_tag = "no_tag"
                    else:
                        nn_tag = sentence[i + 2][1]
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0], i - 2, n, sentence[i + 1][1], nn_tag)

                    self.histories.append(history)


def location_in_sen(i, n):
    """
    classifies the location of the word in the sentence
    :param i: index of word
    :param n: length of sentence
    """
    if n > 20:
        len_n = "long"
    else:
        len_n = "short"
    if i == 0:
        return ("start", len_n)
    if i < 5:
        return (str(i), len_n)

    if i == n - 2:
        return ("pend", len_n)
    if i == n - 1:
        return ("end", len_n)
    else:
        return (str(round((i / n) * 10) / 10), len_n)


def number_or_word(word):
    """

    :param word: word
    :return: number that represents the type of the word
    """
    if any(char.isalpha() for char in word) and any(char.isdigit() for char in word):
        return 1
    elif all(char.isdigit() or char in [".", "-", ",", "+"] for char in word):
        return 2
    elif all(char.isalpha() for char in word):
        return 4
    else:
        return 0


def get_prefix_suffix_list(word):
    """
    return all the suffixes and prefixes of the given word up to 6 letters from each direction
    """
    prefix_list = []
    suffix_list = []
    if len(word) >= 1:
        prefix_list.append(word[:1])
        suffix_list.append(word[-1:])
    if len(word) >= 2:
        prefix_list.append(word[:2])
        suffix_list.append(word[-2:])
    if len(word) >= 3:
        prefix_list.append(word[:3])
        suffix_list.append(word[-3:])
    if len(word) >= 4:
        prefix_list.append(word[:4])
        suffix_list.append(word[-4:])
    if len(word) >= 5:
        prefix_list.append(word[:5])
        suffix_list.append(word[-5:])
    if len(word) >= 6:
        prefix_list.append(word[:6])
        suffix_list.append(word[-6:])

    for i in range(1, 4):
        if len(word) == i:
            prefix_list.append((word, "full"))

    return prefix_list, suffix_list


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            "f100": OrderedDict(),
            "f101": OrderedDict(),
            "f102": OrderedDict(),
            "f103": OrderedDict(),
            "f104": OrderedDict(),
            "f105": OrderedDict(),
            "f106": OrderedDict(),
            "f107": OrderedDict(),
            "f108": OrderedDict(),
            "f108.2": OrderedDict(),
            "f109": OrderedDict(),
            "f110": OrderedDict(),
            "f111": OrderedDict(),
            "f112": OrderedDict(),
            "f113": OrderedDict(),
            "f114": OrderedDict(),
            "f115": OrderedDict(),
            "f116": OrderedDict(),
        }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold[feat_class]:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1

        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        small_matrix - the i'th row is the feature representation of the i'th word.
        big_matrix - in rows [|tags|*i, |tags|*(i+1)] (~+-1) we have the i'th word in the text with
        every tag. Not completely true because some combinations of (word, tag) might not have passed
        the threshold so they don't have features.
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (
                hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8], hist[9], hist[10])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    p_word = history[2]
    p_tag = history[3]
    pp_word = history[4]
    pp_tag = history[5]
    n_word = history[6]
    word_index = history[7]
    sen_len = history[8]
    n_tag = history[9]
    nn_tag = history[10]
    features = []

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # f101, f102
    prefix_list, suffix_list = get_prefix_suffix_list(c_word)
    for curr_pre, curr_suf in zip(prefix_list, suffix_list):
        if (curr_pre, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(curr_pre, c_tag)])
        if (curr_suf, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(curr_suf, c_tag)])

    # f103, f104, f105
    if (pp_tag, p_tag, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(pp_tag, p_tag, c_tag)])
    if (p_tag, c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(p_tag, c_tag)])
    if (c_tag) in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][(c_tag)])

    # f106, f107
    if (p_word, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(p_word, c_tag)])
    if (n_word, c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(n_word, c_tag)])

    # f108
    curr_capital_count = (str(min(sum(1 for char in c_word if char.isupper()), 7)), len(c_word))
    p_capital_count = (str(min(sum(1 for char in p_word if char.isupper()), 7)), len(p_word))
    if (curr_capital_count, c_tag) in dict_of_dicts["f108"]:
        features.append(dict_of_dicts["f108"][(curr_capital_count, c_tag)])
    if (p_capital_count, c_tag) in dict_of_dicts["f108.2"]:
        features.append(dict_of_dicts["f108.2"][(p_capital_count, c_tag)])

    # f109
    type_word = str(number_or_word(c_word))
    if (type_word, c_tag) in dict_of_dicts["f109"]:
        features.append(dict_of_dicts["f109"][(type_word, c_tag)])

    # f110, f111
    p_word_type = str(number_or_word(p_word))
    pp_word_type = str(number_or_word(pp_word))
    if (p_word_type, c_tag) in dict_of_dicts["f110"]:
        features.append(dict_of_dicts["f110"][(p_word_type, c_tag)])
    if (pp_word_type, c_tag) in dict_of_dicts["f111"]:
        features.append(dict_of_dicts["f111"][(pp_word_type, c_tag)])

    # f112
    word_loc = location_in_sen(word_index, sen_len)
    if (word_loc, c_tag) in dict_of_dicts["f112"]:
        features.append(dict_of_dicts["f112"][(word_loc, c_tag)])

    # f113
    chars = ["-", ",", "$", "%", "!", "?", "...", "--", ":", ";"]
    if (max(len(c_word), 12), c_tag) in dict_of_dicts["f113"]:
        features.append(dict_of_dicts["f113"][(max(len(c_word), 12), c_tag)])
    for char in chars:
        if char in c_word:
            if (char, len(c_word), c_tag) in dict_of_dicts["f113"]:
                features.append(dict_of_dicts["f113"][(char, len(c_word), c_tag)])

    # f114
    if c_word[0].isupper() and not c_word[-1].isupper():
        if (len(c_word), c_tag) in dict_of_dicts["f114"]:
            features.append(dict_of_dicts["f114"][(len(c_word), c_tag)])
        if (c_tag) in dict_of_dicts["f114"]:
            features.append(dict_of_dicts["f114"][(c_tag)])

    # f115
    if (c_word, c_tag, n_tag) in dict_of_dicts["f115"]:
        features.append(dict_of_dicts["f115"][(c_word, c_tag, n_tag)])
    if (p_tag, c_tag, n_tag) in dict_of_dicts["f115"]:
        features.append(dict_of_dicts["f115"][(p_tag, c_tag, n_tag)])
    if (pp_tag, p_tag, c_tag, n_tag) in dict_of_dicts["f115"]:
        features.append(dict_of_dicts["f115"][(pp_tag, p_tag, c_tag, n_tag)])
    if (c_tag, n_tag) in dict_of_dicts["f115"]:
        features.append(dict_of_dicts["f115"][(c_tag, n_tag)])
    if (number_or_word(c_word), n_tag) in dict_of_dicts["f115"]:
        features.append(dict_of_dicts["f115"][(number_or_word(c_word), n_tag)])

    # f116
    if (c_tag, n_tag, nn_tag) in dict_of_dicts["f116"]:
        features.append(dict_of_dicts["f116"][(c_tag, n_tag, nn_tag)])
    if (p_tag, c_tag, n_tag, nn_tag) in dict_of_dicts["f116"]:
        features.append(dict_of_dicts["f116"][(p_tag, c_tag, n_tag, nn_tag)])
    if (number_or_word(c_word), n_tag, nn_tag) in dict_of_dicts["f116"]:
        features.append(dict_of_dicts["f116"][(number_or_word(c_word), n_tag, nn_tag)])

    return features


def preprocess_train(train_path, threshold):
    """
    :param train_path: path to train file
    :param threshold: dictionary of threashold for each feature
    :return: statistics and featrue2id objects that are exctracted based on the train file
    """
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))

    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
