from main import read_file_to_list
from main import read_parsed_data
from main import run_svm
from main import run_dec_tree
from main import run_nb
from main import make_top_words_list
import numpy as np
import matplotlib.pyplot as plt

CLASS_NATIVE_LABEL = "native"
CLASS_NON_NATIVE_LABEL = "non-native"

PARSED_DATA_FULL_PATH = "../parsedData/alldata45.txt"
# PARSED_DATA_FULL_PATH = "../parsedData/shortalldata.txt"
FUNCTION_WORDS_FILE = "../parsedData/functionWords.txt"
RANDOMIZE_DATA = False  # will alter the train-test samples
CLASS_NATIVE_VALUE = 1
CLASS_NON_NATIVE_VALUE = -1
TRAIN_TEST_SPLIT = 0.8

FUNC_WORDS = True
TOP_WORDS = True
NUM_OF_TOP_WORDS = 230

RUN_SVM_K_FOLD = True
RUN_DEC_TREE = True
RUN_NB = True


def print_top_x_f_words(class_name, super_vec, func_words, x=10):
    temp_super_vec = super_vec[:]
    total = 0
    print(class_name)
    for i in range(x):
        index_max = np.argmax(temp_super_vec)
        print("{}){} {}".format(i, func_words[index_max], temp_super_vec[index_max]))
        total += temp_super_vec[index_max]
        temp_super_vec[index_max] = -1
    print("Total count for 10 top f_words={}".format(total))
    return


def print_most_x_far_words(nat_super_vec, non_super_vec, func_words, x=10):
    temp_nat_super_vec = nat_super_vec[:]
    temp_non_super_vec = non_super_vec[:]
    print("Finding top {} words that are most different in usage...".format(x))
    seen_words = []
    for j in range(x):
        max_diff = 0
        for i in range(len(func_words)):
            diff = abs(temp_nat_super_vec[i] - temp_non_super_vec[i])
            if func_words[i] not in seen_words and diff > max_diff:
                max_diff, word, nat_count, non_count = diff, func_words[i], temp_nat_super_vec[i], temp_non_super_vec[i]
        seen_words.append(word)
        msg = "{})Word(s)='{}'. Native count={} vs Non-native count={}. Diff={}"
        print(msg.format(j+1, word, nat_count, non_count, max_diff))
    print(seen_words)
    return


def make_super_vec(f_len, data_x):  # get 1 feature vector in the size of f_words that is a sum of all samples f_vecs
    super_vec = [0] * f_len
    for i in range(len(data_x)):
        super_vec = [x + y for x, y in zip(super_vec, data_x[i])]
    return super_vec


def split_all_data_to_classes(data_x, data_y):
    data_x_nat = []
    data_x_non = []
    for i in range(len(data_y)):
        if data_y[i] == CLASS_NATIVE_VALUE:
            data_x_nat.append(data_x[i])
        else:
            data_x_non.append(data_x[i])
    return data_x_nat, data_x_non


def show2bar_charts(nat_super_vec, non_super_vec):
    plt.subplot(121)
    y_pos = np.arange(len(nat_super_vec))
    plt.bar(x=y_pos, height=nat_super_vec)
    plt.ylabel('#times')
    plt.xlabel('function words indices')
    plt.title("{}: usage of function words".format("Native:"))

    plt.subplot(122)
    y_pos = np.arange(len(non_super_vec))
    plt.bar(x=y_pos, height=non_super_vec)
    plt.ylabel('#times')
    plt.xlabel('function words indices')
    plt.title("{}: usage of function words".format("NON-Native:"))

    plt.show()
    return


def calc_mse_from_avg(name, super_vec):
    sv = super_vec[:]
    list_sum = sum(sv)
    list_len = len(sv)
    list_avg = list_sum/list_len
    nat_mse = 0
    for i in range(list_len):
        nat_mse += pow(float(sv[i]) - list_avg, 2)

    nat_mse /= list_len
    print("{}: average usage={}. mse from avg={}".format(name, list_avg, nat_mse))
    return


def fix_test_size_and_change_train_size(data_x, data_y, test_fixed_size, train_size_start, clf="SVM", jumps=1500):
    split_index = len(data_x) - test_fixed_size

    test_x_fixed = data_x[split_index:]
    test_y_fixed = data_y[split_index:]

    remaining_data_x = data_x[:split_index]
    remaining_data_y = data_y[:split_index]

    print("{}:".format(clf))
    print("Test data size is fixed to {}".format(len(test_y_fixed)))
    train_size = train_size_start
    i = 1
    while train_size < len(remaining_data_y):  # increasing train size from train_size_start to max
        train_x = remaining_data_x[:train_size]
        train_y = remaining_data_y[:train_size]
        if clf == "SVM":
            score, _, _, fscore_w = run_svm(train_x, train_y, test_x_fixed, test_y_fixed, False)
        elif clf == "DT":
            score, _, _, fscore_w = run_dec_tree(train_x, train_y, test_x_fixed, test_y_fixed, False)
        else:  # NB
            score, _, _, fscore_w = run_nb(train_x, train_y, test_x_fixed, test_y_fixed, False)
        msg = "Attempt {}({}): Train data size={}, Accuracy={:.3f}, fscore={:.3f}"
        print(msg.format(i, clf, len(train_y), float(score), fscore_w))
        train_size += jumps
        i += 1
    train_x = remaining_data_x[:]  # doing last iter on all remaining_data_x
    train_y = remaining_data_y[:]
    if clf == "SVM":
        score, _, _, fscore_w = run_svm(train_x, train_y, test_x_fixed, test_y_fixed, False)
    msg = "Attempt {}({}): Train data size={}, Accuracy={:.3f}, fscore={:.3f}"
    print(msg.format(i, clf, len(train_y), float(score), fscore_w))


def main():
    func_words = read_file_to_list(FUNCTION_WORDS_FILE, -1)
    all_rows = read_file_to_list(PARSED_DATA_FULL_PATH, -1)
    if FUNC_WORDS:
        print("-----------FUNC_WORDS_START---------------------------")
        train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy = read_parsed_data(all_rows, func_words)
        data_x = train_x_svm_rdy + test_x_svm_rdy
        data_y = train_y_svm_rdy + test_y_svm_rdy

        data_x_nat, data_x_non = split_all_data_to_classes(data_x, data_y)

        nat_super_vec = make_super_vec(len(func_words), data_x_nat)
        non_super_vec = make_super_vec(len(func_words), data_x_non)

        # print_top_x_f_words(CLASS_NATIVE_LABEL, nat_super_vec, func_words, 10)
        # print_top_x_f_words(CLASS_NON_NATIVE_LABEL, non_super_vec, func_words, 10)

        print_most_x_far_words(nat_super_vec, non_super_vec, func_words, 10)

        # # show2bar_charts(nat_super_vec,non_super_vec)
        # print("\nMSE from count average:")
        # calc_mse_from_avg("Native", nat_super_vec)
        # calc_mse_from_avg("NON-Native", non_super_vec)

        # fix_test_size_and_change_train_size(data_x, data_y, 1000, 2500, "SVM", 2000)
        # fix_test_size_and_change_train_size(data_x, data_y, 1000, 2500, "DT", 2000)
        # fix_test_size_and_change_train_size(data_x, data_y, 1000, 2500, "NB", 2000)
        print("-------------FUNC_WORDS_END-------------------------")
    if TOP_WORDS:
        top_words_list = make_top_words_list(all_rows, func_words)
        # loads parsed data as count words array
        train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy = read_parsed_data(all_rows, top_words_list)
        data_x = train_x_svm_rdy + test_x_svm_rdy
        data_y = train_y_svm_rdy + test_y_svm_rdy

        data_x_nat, data_x_non = split_all_data_to_classes(data_x, data_y)

        nat_super_vec = make_super_vec(len(top_words_list), data_x_nat)
        non_super_vec = make_super_vec(len(top_words_list), data_x_non)

        print("------------------TOP_WORDS_START--------------------")
        print_top_x_f_words(CLASS_NATIVE_LABEL, nat_super_vec, top_words_list, 10)
        print_top_x_f_words(CLASS_NON_NATIVE_LABEL, non_super_vec, top_words_list, 10)
        print_most_x_far_words(nat_super_vec, non_super_vec, top_words_list, 10)
        # fix_test_size_and_change_train_size(data_x, data_y, 1000, 2500, "SVM", 2000)
        # fix_test_size_and_change_train_size(data_x, data_y, 1000, 2500, "DT", 2000)
        # fix_test_size_and_change_train_size(data_x, data_y, 1000, 2500, "NB", 2000)
        print("---------------TOP_WORDS_END-----------------------")

    return


if __name__ == "__main__":
    main()
