import os
import numpy as np
import math
from sklearn.svm import SVC
from time import time
from collections import Counter
import matplotlib.pyplot as plt

RAW_DATA_PATH = "../rawData/reddit/"
NON_NATIVE_RAW_FOLDER_NAME = "non-native/"
NATIVE_RAW_FOLDER_NAME = "native/"
RAW_DATA_SIZE_FOR_EACH_CLASS = 1000000
CLASS_NATIVE_LABEL = "native"
CLASS_NON_NATIVE_LABEL = "non-native"
MINIMUM_ROW_LENGTH = 35

PARSED_DATA_FULL_PATH = "../parsedData/alldata.txt"
FUNCTION_WORDS_FILE = "../parsedData/functionWords.txt"
RANDOMIZE_DATA = False  # will alter the train-test samples
DATA_SET_SIZE = 150000
CLASS_NATIVE_VALUE = 1
CLASS_NON_NATIVE_VALUE = -1
TRAIN_TEST_SPLIT = 0.8


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
        if len(line) >= MINIMUM_ROW_LENGTH:
            line = ("[" + label + "] " + line)
            my_list.append(line.lower())
    return my_list


def parse_class(label, base_path, file_names):
    size = RAW_DATA_SIZE_FOR_EACH_CLASS
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
        base_path = RAW_DATA_PATH + NATIVE_RAW_FOLDER_NAME
        native_file_names = os.listdir(base_path)
        native_data = parse_class(CLASS_NATIVE_LABEL, base_path, native_file_names)

        base_path = RAW_DATA_PATH + NON_NATIVE_RAW_FOLDER_NAME
        non_native_file_names = os.listdir(base_path)
        non_native_data = parse_class(CLASS_NON_NATIVE_LABEL, base_path, non_native_file_names)

        all_semi_raw_data = native_data + non_native_data
        np.random.shuffle(all_semi_raw_data)
        print("    all_data size is {}".format(len(all_semi_raw_data)))
        print("    saving to file: {}".format(PARSED_DATA_FULL_PATH))
        dst_file = open(PARSED_DATA_FULL_PATH, 'w+', encoding="utf8")
        for line in all_semi_raw_data:
            dst_file.write(line + '\n')

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
        if line not in my_list:
            my_list.append(line.lower())
    print("    Reading file({} lines) {}".format(len(my_list), file))
    return my_list


def get_data(train_test_split):
    all_rows = read_file_to_list(PARSED_DATA_FULL_PATH, DATA_SET_SIZE)
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
        # TODO - separate cases - if func_word == 1 go over sample. if bigger then do find
        for func_word in func_words:
            if len(func_word.split()) > 1:  # func_word is a phrase
                func_word_split = func_word.split()
                sample_words = sample.split()
                word_counter = len(sample_words)
                print(func_word_split)
                print(sample_words)
                print(word_counter)
                for i in range(word_counter):
                    if sample_words[i] == func_word_split[0]:
                        match = True
                        for j in range(1, len(func_word_split)):
                            if sample_words[i + j] != func_word_split[j]:
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


def read_parsed_data():
    print("Reading parsed data")
    func_words = read_file_to_list(FUNCTION_WORDS_FILE, -1)
    train_x, train_y, test_x, test_y = get_data(TRAIN_TEST_SPLIT)

    train_x_svm_ready = build_feature_vector(func_words, train_x, "Train")
    train_y_svm_ready = enumerate_labels(train_y, "Train")

    test_x_svm_ready = build_feature_vector(func_words, test_x, "Test")
    test_y_svm_ready = enumerate_labels(test_y, "Test")

    return train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready


def run_svm(train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready):
    print("Running SVM...")
    clf = SVC()
    specs = clf.fit(train_x_svm_ready, train_y_svm_ready)
    print("SVM info:")
    print("  {}".format(specs))
    score = clf.score(test_x_svm_ready, test_y_svm_ready)
    print("    Accuracy={}%".format(score*100))

    return


def run_example():
    print("Running SVM toy example...")
    train_x = np.array([[-3], [-2], [2], [3]])
    train_y = np.array([-1, -1, 1, 1])
    clf = SVC()
    specs = clf.fit(train_x, train_y)
    print("SVM info:")
    print("  {}".format(specs))
    test_x = [[5], [-1], [3], [2], [-0.1], [-2]]
    test_y = [1, -1, 1, 1, -1, -1]
    score = clf.score(test_x, test_y)
    print("Accuracy={}%".format(score*100))
    return


def main():
    start_time = time()

    maybe_parse_data()
    train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready = read_parsed_data()
    run_svm(train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready)
    # run_example()

    duration = time() - start_time
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("duration(formatted HH:MM:SS): {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))



if __name__ == "__main__":
    main()
