import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def main():
    threshold = {
        "f100": 1,
        "f101": 1,
        "f102": 1,
        "f103": 1,
        "f104": 1,
        "f105": 1,
        "f106": 1,
        "f107": 1,
        "f108": 1,
        "f108.2": 1,
        "f109": 1,
        "f110": 1,
        "f111": 1,
        "f112": 1,
        "f113": 1,
        "f114": 1,
        "f115": 1,
        "f116": 1,
    }

    # generate m1_weights.pkl
    lam = 0.5
    train_path = "data/train1.wtag"
    weights_path = 'm1_weights.pkl'
    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    # generate comp1.words
    test_path = "data/comp1.words"
    predictions_path = "comp_m1.wtag"
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)

    # generate m2_weights.pkl
    lam = 0.01
    train_path = "data/train2.wtag"
    weights_path = 'm2_weights.pkl'
    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    # generate comp2.words
    test_path = "data/comp2.words"
    predictions_path = "comp_m2.wtag"
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


if __name__ == '__main__':
    main()
