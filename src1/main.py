import os
import numpy as np
import math
import shutil
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random


RAW_DATA_PATH = "../rawData/reddit/"
NON_NATIVE_RAW_FOLDER_NAME = "non-native/"
NATIVE_RAW_FOLDER_NAME = "native/"
TRAIN_DATA_SIZE_FOR_EACH_CLASS = 10000
MINIMUM_ROW_LENGTH = 7

PARSED_DATA_PATH = "../parsedData/data/"
FUNCTION_WORDS_FILE = "../parsedData/functionWords.txt"
NON_NATIVE_FILENAME = "non-native.txt"
NATIVE_FILENAME = "native.txt"
TRAIN_TEST_SPLIT = 0.8


def create_parsed_file(base_path, files, dst_file_name, dst_file_size, sentence_length):
    global_sentence_array = []
    line_from_each_file = math.floor(dst_file_size/len(files))
    print("line_from_each_file: {} / {}  ={}".format(dst_file_size, len(files), line_from_each_file))
    for cur_file in files:
        print("cur_file= {}".format(cur_file))
        f = open(base_path + cur_file, 'r', encoding="utf8")
        sentence_array = []
        for line in f:
            line = remove_brackets(line)
            line = line.strip()
            if len(line) >= sentence_length:
                sentence_array.append(line.lower())

        np.random.shuffle(sentence_array)

        global_sentence_array += sentence_array[:line_from_each_file]
        print("sentence_array size {}".format(len(sentence_array)))
        print("global_sentence_array size {}".format(len(global_sentence_array)))
        if len(global_sentence_array) >= 1500:  # TODO delete
            break

    # dump array to file
    dst_file = open(dst_file_name, 'w+', encoding="utf8")
    for line in global_sentence_array:
        dst_file.write(line + '\n')

    print("file {} with {} lines was created".format(dst_file_name, len(global_sentence_array)))


def remove_brackets(sentence):
    sub_string = sentence[sentence.find('['):sentence.find(']')+1]
    sentence = sentence.replace(sub_string, '', 1)
    sub_string = sentence[sentence.find('['):sentence.find(']')+1]
    sentence = sentence.replace(sub_string, '', 1)
    return sentence


def parse_data():
    print("parsing raw data:")
    row_len = MINIMUM_ROW_LENGTH
    samples_size = TRAIN_DATA_SIZE_FOR_EACH_CLASS

    base_path = RAW_DATA_PATH + NATIVE_RAW_FOLDER_NAME
    native_file_names = os.listdir(base_path)
    create_parsed_file(base_path, native_file_names, PARSED_DATA_PATH + NATIVE_FILENAME, samples_size, row_len)

    base_path = RAW_DATA_PATH + NON_NATIVE_RAW_FOLDER_NAME
    non_native_file_names = os.listdir(RAW_DATA_PATH + NON_NATIVE_RAW_FOLDER_NAME)
    create_parsed_file(base_path, non_native_file_names, PARSED_DATA_PATH + NON_NATIVE_FILENAME, samples_size, row_len)


def maybe_parse_data():
    if not os.path.exists(PARSED_DATA_PATH):
        print("data does not exists")
        os.makedirs(PARSED_DATA_PATH)
        parse_data()
    else:
        print("data exists")


def read_file_to_list(file, name):
    my_list = []
    f = open(file, 'r', encoding="utf8")
    for line in f:
        line = line.strip()
        if line not in my_list:
            my_list.append(line)

    print("{} len is {}".format(name, len(my_list)))
    return my_list


def get_data(train_test_split):

    data_x_native = read_file_to_list(PARSED_DATA_PATH + NATIVE_FILENAME, "data_x_native")
    data_y_native = [1] * len(data_x_native)
    data_x_non_native = read_file_to_list(PARSED_DATA_PATH + NON_NATIVE_FILENAME, "data_x_non_native")
    data_y_non_native = [2] * len(data_x_non_native)

    all_data_x = data_x_native + data_x_non_native
    all_data_y = data_y_native + data_y_non_native

    c = list(zip(all_data_x, all_data_y))
    random.shuffle(c)
    temp_x_2, temp_y_2 = zip(*c)
    all_data_x = temp_x_2
    all_data_y = temp_y_2

    split_ind = math.floor(len(all_data_x) * train_test_split)
    train_x = all_data_x[:split_ind]
    train_y = all_data_y[:split_ind]
    test_x = all_data_x[split_ind:-1]
    test_y = all_data_y[:split_ind:-1]
    print("train set size is {}".format(len(train_x)))
    print("test set size is {}".format(len(test_x)))
    return train_x, train_y, test_x, test_y


def build_feature_vector(func_words, data_x):
    all_f_vecs = []
    for sample in data_x:
        f_vec = dict.fromkeys(func_words, 0)
        # print("{}) {}".format( sample))
        for func_word in f_vec.keys():
            # TODO replace with regex - no [a-z] before and after the word
            a = sample.find(func_word)
            if a != -1:
                # checking that the func word is not contained in another word (ex: the 'a' is not part of 'gilad')
                pre = a > 0 and not sample[a-1].isalpha()  # checking before the word there isn't another char
                post = a + len(func_word) < len(sample) and not sample[a + len(func_word)].isalpha()  # checking after
                if pre and post:
                    f_vec[func_word] += 1
        all_f_vecs.append(f_vec)

    print("all_f_vecs size is {}".format(len(all_f_vecs)))
    return all_f_vecs


def build_feature_vector2(func_words, data_x):
    all_f_vecs = []
    iters = 0
    for sample in data_x:
        f_vec = [0] * len(func_words)
        count = 0
        for func_word in func_words:
            # TODO replace with regex - no [a-z] before and after the word
            a = sample.find(func_word)
            if a != -1:
                # checking that the func word is not contained in another word (ex: the 'a' is not part of 'gilad')
                pre = a > 0 and not sample[a-1].isalpha()  # checking before the word there isn't another char
                post = a + len(func_word) < len(sample) and not sample[a + len(func_word)].isalpha()  # checking after
                if pre and post:
                    f_vec[count] += 1
            count += 1
        iters += 1
        all_f_vecs.append(f_vec)

    print("all_f_vecs size is {}".format(len(all_f_vecs)))
    return all_f_vecs


def print_graph(x, title, xlabel, ylabel, func_a_data, func_b_data):
    plt.subplot(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(func_a_data, 'r')
    plt.plot(func_b_data, 'g')


def main():
    # if os.path.exists(PARSED_DATA_PATH): shutil.rmtree(PARSED_DATA_PATH)  # TODO delete
    maybe_parse_data()
    func_words = read_file_to_list(FUNCTION_WORDS_FILE, "func_words")
    train_x, train_y, test_x, test_y = get_data(TRAIN_TEST_SPLIT)
    # train_x_fv = build_feature_vector(func_words, train_x)
    train_x_fv = build_feature_vector2(func_words, train_x)
    test_x_fv = build_feature_vector2(func_words, test_x)

    # working code
    # X = np.array([[-1], [-2], [1], [5]])
    # y = np.array([1, 1, 2, 2])
    # clf = SVC()
    # clf.fit(X, y)
    # print(clf.predict([[5], [-1], [3], [2], [0], [-2]]))

    clf = SVC()
    clf.fit(train_x_fv, train_y)
    score = clf.score(test_x_fv, test_y)
    print("acc={}".format(score))

    prediction_y = clf.predict(test_x_fv)
    good = 0
    for i in range(len(test_y)):
        if test_y[i] == prediction_y[i]:
            good += 1
    acc = good / len(test_y)
    print(acc)

    # # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    # #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    # #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    # #     tol=0.001, verbose=False)

if __name__ == "__main__":
    main()
