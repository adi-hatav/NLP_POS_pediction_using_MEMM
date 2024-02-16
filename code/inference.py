import scipy
import scipy as sp
from preprocessing import read_test
from tqdm import tqdm
import numpy as np
from preprocessing import represent_input_with_features


def memm_viterbi(sentence, pre_trained_weights, feature2id, our_pred):
    """
    viterbi algorithm for tagging a sentence
    :param sentence: the sentence to tag. A list of words starting with * * and ending with ~
    :param pre_trained_weights: pre trained weights for the calculations of q from viterbi algorithm
    :param feature2id: feature2id object
    :param our_pred: list of tags from our last prediction (we run viterbi a few times so we can have features that are
    based on tags of future words.
    :return: a list with tags corresponding to the given sentence
    """
    n = len(sentence) - 2
    tags = feature2id.feature_statistics.tags
    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    index_to_tag = {index: tag for index, tag in enumerate(tags)}

    probs_matrix = np.zeros((n, len(tags), len(tags)), dtype=float)
    args_matrix = np.zeros((n, len(tags), len(tags)), dtype=int)
    start_index = tag_to_index['*']
    probs_matrix[0][start_index][start_index] = 1

    for i in range(1, n):
        prev_probs_mat = probs_matrix[i - 1]
        pp_tags, p_tags, row_sums = beam_search_args(i, start_index, prev_probs_mat, 1)
        for pp_tag in pp_tags:
            prev_probs_mat[pp_tag, :] /= row_sums
        for last_tag in range(len(tags)):
            for curr_tag in range(len(tags)):
                if last_tag in p_tags:
                    prob, arg = calc_best_prob(our_pred, n - 1, pp_tags, start_index, prev_probs_mat, tags,
                                               index_to_tag,
                                               sentence, pre_trained_weights, feature2id, i + 1, curr_tag, last_tag)
                    probs_matrix[i][last_tag][curr_tag] = prob
                    args_matrix[i][last_tag][curr_tag] = arg

    pred_tags = []
    # Get the indices of the maximum value
    max_index = np.argmax(probs_matrix[n - 1])
    # Convert the flattened index to 2D indices
    max_indices = np.unravel_index(max_index, probs_matrix[n - 1].shape)
    curr_tag = max_indices[1]
    p_tag = max_indices[0]
    pred_tags.append(index_to_tag[curr_tag])
    pred_tags.append(index_to_tag[p_tag])

    for i in range(n - 1, 2, -1):
        temp = args_matrix[i][p_tag][curr_tag]
        pred_tags.append(index_to_tag[temp])
        curr_tag = p_tag
        p_tag = temp

    return pred_tags[::-1]


def calc_best_prob(our_pred, n, pp_tags, start_index, prev_mat, tags, index_to_tag, sentence, pre_trained_weights,
                   feature2id,
                   i, curr_tag, last_tag):
    """
    calculate q for viterbi
    :param our_pred: list of tags from our last prediction (we run viterbi a few times so we can have features that are
    based on tags of future words.
    :param n: length of the sentence
    :param pp_tags: rows to explore
    :param start_index: the index of the * tag
    :param prev_mat: pi(i-1)
    :param tags: set of all the tags seen at training
    :param index_to_tag: dictionary for converting index to tag
    :param sentence: list of words
    :param pre_trained_weights: self-explanatory
    :param feature2id: Feature2Id object
    :param i: the index of the current word in the sentence
    :param curr_tag: the tag we assign to the current word
    :param last_tag: the tag we assign to the last word
    :return: the max and argmax probability across all possible assignments of the tag of the word i-2
    """
    if i == 2 or i == 3:
        possible_tags = [start_index]
    else:
        possible_tags = pp_tags

    temp_array = np.zeros(len(tags))
    best_prob = 0
    best_arg = 0
    nn_tag = "no_tag"
    next_tag = "none"
    if not our_pred is None:
        next_tag = our_pred[i + 1]
        if i + 2 < len(our_pred):
            nn_tag = our_pred[i + 2]
    for tag in possible_tags:
        history = (sentence[i], index_to_tag[curr_tag], sentence[i - 1], index_to_tag[last_tag], sentence[i - 2],
                   index_to_tag[tag], sentence[i + 1], i - 2, n, next_tag, nn_tag)

        features_representation = represent_input_with_features(history, feature2id.feature_to_idx)

        prob = np.exp(np.sum(np.array([pre_trained_weights[feature] for feature in features_representation])))
        temp_array[tag] = prob
        if prev_mat[tag][last_tag] == 0:
            continue
        prob = prob * prev_mat[tag][last_tag]
        if prob > best_prob:
            best_prob = prob
            best_arg = tag

    return best_prob, best_arg


def beam_search_args(i, start_index, probs_matrix, b):
    """
    :param i: the index of the matrix
    :param start_index: the index of the tag of *
    :param probs_matrix: pi[i] from the lecture
    :param b: the beam search parameter
    """
    if i == 1:
        return [start_index], [start_index], 1

    # Find top b max-sum columns
    column_sums = np.sum(probs_matrix, axis=0)
    top_column_indices = np.argsort(-column_sums)[:b]

    # Find top b max-sum rows
    row_sums = np.sum(probs_matrix, axis=1)
    top_row_indices = np.argsort(-row_sums)[:b]
    row_sums = np.sum(row_sums[top_row_indices])

    if i == 2:
        return [start_index], top_column_indices, row_sums

    return top_row_indices, top_column_indices, row_sums


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    """
    :param test_path: path to test file
    :param pre_trained_weights: pre trained weights for the calculations of q from viterbi algorithm
    :param feature2id: feature2id object
    :param predictions_path: output file path
    :return: tags the test and outputs to the predictions_path file
    """
    tagged = "test" in test_path or "train" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "w")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        tags = sen[1]
        tags = tags[2:-1]
        our_pred = None
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, our_pred)
        pred = ["*", "*"] + pred + ["~"]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, pred)
        pred = ["*", "*"] + pred + ["~"]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, pred)
        pred = ["*", "*"] + pred + ["~"]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, pred)
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
