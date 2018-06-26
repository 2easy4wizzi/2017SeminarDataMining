from main import read_file_to_list
from main import read_parsed_data
from main import run_svm
import numpy as np
import matplotlib.pyplot as plt

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
K = 3
RUN_DEC_TREE = True
RUN_NB = True


def print_top_x_f_words(class_name, super_vec, func_words):
    temp_super_vec = super_vec[:]
    total = 0
    print(class_name)
    for i in range(10):
        index_max = np.argmax(temp_super_vec)
        print("{}){} {}".format(i, func_words[index_max], temp_super_vec[index_max]))
        total += temp_super_vec[index_max]
        temp_super_vec[index_max] = -1
    print("Total count for 10 top f_words={}".format(total))
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


def fix_test_size_and_change_train_size(data_x, data_y, test_fixed_size, train_size_start):
    split_index = len(data_x) - test_fixed_size

    test_x_fixed = data_x[split_index:]
    test_y_fixed = data_y[split_index:]

    remaining_data_x = data_x[:split_index]
    remaining_data_y = data_y[:split_index]

    print(len(data_x))
    print(len(remaining_data_x))
    print(len(test_x_fixed))

    print("Test data size is fixed to {}".format(len(test_y_fixed)))
    train_size = train_size_start
    i = 1
    while train_size < len(remaining_data_y):
        train_x = remaining_data_x[:train_size]
        train_y = remaining_data_y[:train_size]
        print("Attempt {}: Train data size set to {}".format(i, len(train_y)))
        run_svm(train_x, train_y, test_x_fixed, test_y_fixed)
        train_size += 1500
        i += 1
    train_x = remaining_data_x[:]
    train_y = remaining_data_y[:]
    print("Attempt {}: Train data size set to {}".format(i, len(train_y)))
    run_svm(train_x, train_y, test_x_fixed, test_y_fixed)


def main():
    func_words = read_file_to_list(FUNCTION_WORDS_FILE, -1)
    all_rows = read_file_to_list(PARSED_DATA_FULL_PATH, -1)
    train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy = read_parsed_data(all_rows, func_words)
    data_x = train_x_svm_rdy + test_x_svm_rdy
    data_y = train_y_svm_rdy + test_y_svm_rdy

    # data_x_nat, data_x_non = split_all_data_to_classes(data_x, data_y)
    #
    # nat_super_vec = make_super_vec(len(func_words), data_x_nat)
    # non_super_vec = make_super_vec(len(func_words), data_x_non)
    #
    # print_top_x_f_words("Native:", nat_super_vec, func_words)
    # print_top_x_f_words("NON-Native:", non_super_vec, func_words)
    #
    # # show2bar_charts(nat_super_vec,non_super_vec)
    # print("\nMSE from count average:")
    # calc_mse_from_avg("Native", nat_super_vec)
    # calc_mse_from_avg("NON-Native", non_super_vec)

    fix_test_size_and_change_train_size(data_x, data_y, 1000, 2000)

    return


if __name__ == "__main__":
    main()
