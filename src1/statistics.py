from main import read_file_to_list
from main import read_parsed_data

PARSED_DATA_FULL_PATH = "../parsedData/alldata.txt"
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


def main():
    func_words = read_file_to_list(FUNCTION_WORDS_FILE, -1)
    all_rows = read_file_to_list(PARSED_DATA_FULL_PATH, -1)
    train_x_svm_rdy, train_y_svm_rdy, test_x_svm_rdy, test_y_svm_rdy = read_parsed_data(all_rows, func_words)
    data_x = train_x_svm_rdy + test_x_svm_rdy
    data_y = train_y_svm_rdy + test_y_svm_rdy
    data_x_nat = []
    data_x_non = []
    for i in range(len(data_y)):
        if data_y[i] == CLASS_NATIVE_VALUE:
            data_x_nat.append(data_x[i])
        else:
            data_x_non.append(data_x[i])

    f_len = len(func_words)
    nat_len = len(data_x_nat)
    non_len = len(data_x_non)
    nat_super_vec = [0] * f_len
    non_super_vec = [0] * f_len
    for i in range(nat_len):
        nat_super_vec = [x + y for x, y in zip(nat_super_vec, data_x_nat[i])]
    for i in range(non_len):
        non_super_vec = [x + y for x, y in zip(non_super_vec, data_x_non[i])]

    # nat_top_10_words = []
    # nat_top_10_count = []
    # for i in range(10):
    #     index_max = np.argmax(nat_super_vec)
    #     print("{})index_max {} {} {}".format(i, index_max, func_words[index_max], nat_super_vec[index_max]))
    #     nat_top_10_words.append(func_words[index_max])
    #     nat_top_10_count.append(nat_super_vec[index_max])
    #     nat_super_vec[index_max] = -1
    #
    # print(nat_top_10_words)
    # print(nat_top_10_count)
    #
    # non_top_10_words = []
    # non_top_10_count = []
    # for i in range(10):
    #     index_max = np.argmax(non_super_vec)
    #     print("{})index_max {} {} {}".format(i, index_max, func_words[index_max], non_super_vec[index_max]))
    #     non_top_10_words.append(func_words[index_max])
    #     non_top_10_count.append(non_super_vec[index_max])
    #     non_super_vec[index_max] = -1
    #
    # print(non_top_10_words)
    # print(non_top_10_count)

    print(non_super_vec)

    exit(405)



    return


if __name__ == "__main__":
    main()
