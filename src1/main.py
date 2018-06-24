import os
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from time import time
from collections import Counter
import matplotlib.pyplot as plt

RAW_DATA_PATH = "../rawData/"
REDDIT_DIR = "reddit/"
TOFEL_DIR = "toefl/"
NON_NATIVE_RAW_FOLDER_NAME = "non-native/"
NATIVE_RAW_FOLDER_NAME = "native/"
CLASS_NATIVE_LABEL = "native"
CLASS_NON_NATIVE_LABEL = "non-native"
MINIMUM_ROW_LENGTH = 45

PARSED_DATA_FULL_PATH = "../parsedData/alldata.txt"
FUNCTION_WORDS_FILE = "../parsedData/functionWords.txt"
RANDOMIZE_DATA = False  # will alter the train-test samples
CLASS_NATIVE_VALUE = 1
CLASS_NON_NATIVE_VALUE = -1
TRAIN_TEST_SPLIT = 0.8

FEATURE_VECTOR = True
TOP_WORDS = True
NUM_OF_TOP_WORDS = 230

RUN_SVM_K_FOLD = True
K = 7
RUN_DEC_TREE = True
RUN_NB = True


def read_raw_file_to_list(file, max_rows, label):
    my_list = []
    f = open(file, 'r', encoding="utf8")
    for line in f:
        if 0 < max_rows <= len(my_list):  # -1: read all file
            break
        sub_string = line[line.find('['):line.find(']') + 1]
        line = line.replace(sub_string, '', 1)
        sub_string = line[line.find('['):line.find(']') + 1]
        line = line.replace(sub_string, '', 1)
        line = line.strip()
        if len(line.split()) >= MINIMUM_ROW_LENGTH:
            line = ("[" + label + "] " + line)
            my_list.append(line.lower())
    f.close()
    return my_list


def parse_class(label, base_path, file_names, size):
    line_from_each_file = math.floor(size / len(file_names))
    msg = "    There are {} {} files. Total size of samples: floor({}/{})={}"
    print(msg.format(label, len(file_names), size, len(file_names), line_from_each_file))
    all_rows_raw = []
    debug_files_names = ""
    for file in file_names:
        local_list = read_raw_file_to_list(base_path + file, line_from_each_file, label)
        all_rows_raw += local_list
        debug_files_names += (file + " ")
    print("    {} files parsed: {}".format(label, debug_files_names))
    return all_rows_raw


def maybe_parse_data():
    if not os.path.exists(PARSED_DATA_FULL_PATH):
        print("Need to parse raw data")
        print("    Parsing raw data...")

        base_path = RAW_DATA_PATH + TOFEL_DIR
        non_native_file_names = os.listdir(base_path)
        non_native_data = parse_class(CLASS_NON_NATIVE_LABEL, base_path, non_native_file_names, 1000000)

        base_path = RAW_DATA_PATH + REDDIT_DIR + NATIVE_RAW_FOLDER_NAME
        native_file_names = os.listdir(base_path)
        native_data = parse_class(CLASS_NATIVE_LABEL, base_path, native_file_names, len(non_native_data))

        all_semi_raw_data = native_data + non_native_data
        np.random.shuffle(all_semi_raw_data)
        print("    all_data size is {}".format(len(all_semi_raw_data)))
        print("    saving to file: {}".format(PARSED_DATA_FULL_PATH))
        dst_file = open(PARSED_DATA_FULL_PATH, 'w+', encoding="utf8")
        for line in all_semi_raw_data:
            dst_file.write(line + '\n')
        dst_file.close()
        print("    file {} with {} lines was created".format(PARSED_DATA_FULL_PATH, len(all_semi_raw_data)))
        print("    Finished Parsing raw data")
    else:
        print("Parsed data exists")
    return


def read_file_to_list(file, max_rows):
    my_list = []
    f = open(file, 'r', encoding="utf8")
    for line in f:
        if 0 < max_rows <= len(my_list):  # -1: read all file
            break
        line = line.strip()
        my_list.append(line.lower())
    f.close()
    print("    Reading file({} lines) {}".format(len(my_list), file))
    return my_list


def get_data(all_rows, train_test_split):
    if RANDOMIZE_DATA:
        np.random.shuffle(all_rows)  # will affect the train test split
    count_nat = 0
    count_non_nat = 0
    all_data_x = []
    all_data_y = []
    for row in all_rows:
        if row.startswith("[" + CLASS_NATIVE_LABEL):
            all_data_y.append(CLASS_NATIVE_LABEL)
            count_nat += 1
        elif row.startswith("[" + CLASS_NON_NATIVE_LABEL):
            all_data_y.append(CLASS_NON_NATIVE_LABEL)
            count_non_nat += 1
        sub_string = row[row.find('['):row.find(']') + 1]
        row = row.replace(sub_string, '', 1)
        row = row.strip()
        all_data_x.append(row)
    print("    There are {} Native samples and {} Non-Native".format(count_nat, count_non_nat))
    split_ind = math.floor(len(all_data_y) * train_test_split)

    train_x = all_data_x[:split_ind]
    train_y = all_data_y[:split_ind]

    test_x = all_data_x[split_ind:]
    test_y = all_data_y[split_ind:]

    count_nat = 0
    count_non_nat = 0
    for i in range(len(train_y)):
        if train_y[i] == CLASS_NATIVE_LABEL:
            count_nat += 1
        elif train_y[i] == CLASS_NON_NATIVE_LABEL:
            count_non_nat += 1

    print("    Train set size {} - native={}, non-native={}".format(len(train_y), count_nat, count_non_nat))

    count_nat = 0
    count_non_nat = 0
    for i in range(len(test_y)):
        if test_y[i] == CLASS_NATIVE_LABEL:
            count_nat += 1
        elif test_y[i] == CLASS_NON_NATIVE_LABEL:
            count_non_nat += 1
    print("    Test  set size {} - native={}, non-native={}".format(len(test_y), count_nat, count_non_nat))
    return train_x, train_y, test_x, test_y


def build_feature_vector(func_words, data_x, debug_name):
    all_f_vecs = []
    for sample in data_x:
        f_vec = [0] * len(func_words)
        iter_num = 0
        for func_word in func_words:
            if len(func_word.split()) > 1:  # func_word is a phrase
                func_word_split = func_word.split()
                sample_words = sample.split()
                word_counter = len(sample_words)
                for i in range(word_counter):
                    if sample_words[i] == func_word_split[0]:
                        match = True
                        for j in range(1, len(func_word_split)):
                            if (i + j) < len(sample_words) and sample_words[i + j] != func_word_split[j]:
                                match = False
                                break
                        if match:
                            f_vec[iter_num] += 1
            else:
                counts = Counter(sample.split())
                f_vec[iter_num] += counts[func_words[iter_num]]
            iter_num += 1

        all_f_vecs.append(f_vec)

    print("    {} data 'all feature vector' size is {}x{}".format(debug_name, len(all_f_vecs[0]), len(all_f_vecs)))
    return all_f_vecs


def enumerate_labels(data_y, debug_name):
    new_data_y = data_y[:]
    for i in range(len(new_data_y)):
        if new_data_y[i] == CLASS_NATIVE_LABEL:
            new_data_y[i] = CLASS_NATIVE_VALUE
        elif new_data_y[i] == CLASS_NON_NATIVE_LABEL:
            new_data_y[i] = CLASS_NON_NATIVE_VALUE
    print("    {} data size is {}".format(debug_name, len(new_data_y)))
    return new_data_y


def read_parsed_data(all_rows, func_words):  # func_words could be just top x words
    print("Reading parsed data")
    train_x, train_y, test_x, test_y = get_data(all_rows, TRAIN_TEST_SPLIT)

    train_x_svm_ready = build_feature_vector(func_words, train_x, "Train")
    train_y_svm_ready = enumerate_labels(train_y, "Train")

    test_x_svm_ready = build_feature_vector(func_words, test_x, "Test")
    test_y_svm_ready = enumerate_labels(test_y, "Test")

    return train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready


def run_svm(train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready):
    print("Running SVM...")
    clf = SVC(cache_size=7000)
    clf.fit(train_x_svm_ready, train_y_svm_ready)
    # specs = clf.fit(train_x_svm_ready, train_y_svm_ready)
    # print("SVM info:")
    # print("  {}".format(specs))
    score = clf.score(test_x_svm_ready, test_y_svm_ready)
    print("    Accuracy={}%".format(score * 100))
    return


def run_svm_with_k_fold(data_x, data_y, k):
    print("Running K-FOLD SVM with k={}".format(k))
    kf = KFold(n_splits=k)
    all_data_x = np.array(data_x)
    all_data_y = np.array(data_y)
    i = 1
    for train_index, test_index in kf.split(all_data_x):
        print("    split {}".format(i))
        current_train_x, current_test_x = all_data_x[train_index], all_data_x[test_index]
        current_train_y, current_test_y = all_data_y[train_index], all_data_y[test_index]
        run_svm(current_train_x, current_train_y, current_test_x, current_test_y)
        i += 1
    return


def run_dec_tree(train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready):
    print("Running Decision Tree...")
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_x_svm_ready, train_y_svm_ready)
    # specs = clf.fit(train_x_svm_ready, train_y_svm_ready)
    # print("Tree info:")
    # print("  {}".format(specs))
    score = clf.score(test_x_svm_ready, test_y_svm_ready)
    print("    Accuracy={}%".format(score * 100))
    return


def run_nb(train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready):
    print("Running NB...")
    clf = MultinomialNB()
    clf.fit(train_x_svm_ready, train_y_svm_ready)
    # specs = clf.fit(train_x_svm_ready, train_y_svm_ready)
    # print("NB info:")
    # print("  {}".format(specs))
    score = clf.score(test_x_svm_ready, test_y_svm_ready)
    print("    Accuracy={}%".format(score * 100))
    return


# def run_example():
#     print("Running SVM toy example...")
#     train_x = np.array([[-3], [-2], [2], [3]])
#     train_y = np.array([-1, -1, 1, 1])
#     clf = SVC(kernel='linear', cache_size=7000)
#     specs = clf.fit(train_x, train_y)
#     print("SVM info:")
#     print("  {}".format(specs))
#     test_x = [[5], [-1], [3], [2], [-0.1], [-2]]
#     test_y = [1, -1, 1, 1, -1, -1]
#     score = clf.score(test_x, test_y)
#     print("Accuracy={}%".format(score*100))
#     return


def output_all_args(duration):
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("duration(formatted HH:MM:SS): {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
    return


def get_vocabulary_class(all_class, func_words, ignore_list1):
    vocabulary_class = {}
    for row in all_class:
        row_words = row.split()
        for i in range(1, len(row_words)):
            if row_words[i] not in vocabulary_class:
                vocabulary_class[row_words[i]] = 0
            else:
                vocabulary_class[row_words[i]] += 1
    for k in ignore_list1 + func_words:
        vocabulary_class.pop(k, None)

    vocabulary_class = sorted(vocabulary_class.items(), key=lambda kv: kv[1], reverse=True)
    return vocabulary_class


def make_top_words_list(all_rows, func_words):
    all_native = []
    all_non_native = []
    # separate data
    for row in all_rows:
        if row.startswith("[" + CLASS_NATIVE_LABEL):
            all_native.append(row)
        elif row.startswith("[" + CLASS_NON_NATIVE_LABEL):
            all_non_native.append(row)

    ignore_list1 = [',', '.', '``', '\'\'', ')', '(', '*', ']', '[', '**', '\'', '...', '-', '>', '<', ':', '`'
        , '--', '?', ';', '%', '^', ';', '/', "?", '--', '$', '&', '|', '~', '!']

    vocabulary_native = get_vocabulary_class(all_native, func_words, ignore_list1)
    vocabulary_non_native = get_vocabulary_class(all_non_native, func_words, ignore_list1)

    native_top_words_dict = vocabulary_native[:NUM_OF_TOP_WORDS]
    non_native_top_words_dict = vocabulary_non_native[:NUM_OF_TOP_WORDS]

    # str = ""
    # temp = 0
    # for key, value in native_top_words_dict:
    #     str += " " + key
    #     temp += 1
    #     if temp % 5 == 0:
    #         print(str)
    #         str = ""
    # print("XXXXXXXXXXXXXXXXXXXXXXXX")
    # str = ""
    # temp = 0
    # for key, value in non_native_top_words_dict:
    #     str += " " + key
    #     temp += 1
    #     if temp % 5 == 0:
    #         print(str)
    #         str = ""
    # exit(338)
    # combining top words - removing duplicates
    top_words_list = []
    for key, _ in native_top_words_dict:
        if key not in top_words_list:
            top_words_list.append(key)
    for key, _ in non_native_top_words_dict:
        if key not in top_words_list:
            top_words_list.append(key)
    print("    top_words_list size is {}".format(len(top_words_list)))
    return top_words_list


def main():
    # creates parsed data - needs raw data
    maybe_parse_data()
    # reading func_words anyway
    func_words = read_file_to_list(FUNCTION_WORDS_FILE, -1)
    # reading all_rows anyway
    all_rows = read_file_to_list(PARSED_DATA_FULL_PATH, -1)

    if FEATURE_VECTOR:
        print("Starting feature vector classification")
        # loads parsed data as feature vector array
        train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy = read_parsed_data(all_rows, func_words)
        print("-----------FEATURE_VECTOR_START---------------------------")

        if RUN_SVM_K_FOLD:
            for j in range(3, K):
                run_svm_with_k_fold(train_x_svm_rdy + test_x_svm_rdy, train_y_svm_rdy + test_y_svm_rdy, j)

        if RUN_DEC_TREE:
            run_dec_tree(train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy)

        if RUN_NB:
            run_nb(train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy)
        print("-------------FEATURE_VECTOR_END-------------------------")

    if TOP_WORDS:
        print("Starting top x words classification")
        # creates top x words vocabulary. notice, it takes top x from each class but the result isn't 2x(duplicates)
        top_words_list = make_top_words_list(all_rows, func_words)
        # loads parsed data as count words array
        train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy = read_parsed_data(all_rows, top_words_list)
        print("------------------TOP_WORDS_START--------------------")

        if RUN_SVM_K_FOLD:
            for j in range(3, K):
                run_svm_with_k_fold(train_x_svm_rdy + test_x_svm_rdy, train_y_svm_rdy + test_y_svm_rdy, j)

        if RUN_DEC_TREE:
            run_dec_tree(train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy)

        if RUN_NB:
            run_nb(train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy)
        print("---------------TOP_WORDS_END-----------------------")
    return


if __name__ == "__main__":
    start_time = time()
    main()
    output_all_args(time() - start_time)






