from time import time
from ProfileHMM import ProfileHMM
from editdistance import eval

amino_acid: list[str] = list("ARNDCQEGHILKMFPSTWYVBZX")
aligment_table: list[list[str]]
test_case: list[list[str]] = []
hmm: ProfileHMM


def pre_data():
    # 5% random
    # with open("Data/In/data_training.txt") as f_in, open("Data/In/data_training_real.txt", "w") as f_out_train, open("Data/In/data_checking.txt", "w") as f_out_check:
    #     f_out_train.write(f_in.readline())
    #     for line in f_in:
    #         (f_out_check if random() < 0.05 else f_out_train).write(line)

    # 20 test
    # with open("Data/In/data_executing_full.txt") as f_in, open("Data/In/data_checking.txt", "w") as f_out:
    #     for line in f_in:
    #         if random() < 0.2:
    #             f_out.write(line)

    # check -> execute
    # with open("Data/In/data_checking.txt") as f_check, open("Data/In/data_executing.txt", "w") as f_exe:
    #     for line in f_check:
    #         f_exe.write(
    #             "".join(filter(lambda character: character != "-", list(line.strip())))+"\n")

    # random test
    # with open("Condien.txt", "w") as f_rand_test:
    #     for __ in range(5):
    #         f_rand_test.write("".join(random.choice(amino_acid)
    #                           for _ in range(100 + int(random.random() * 100)))
    #                           + "\n")
    pass


def align_from_hidden_path(ref: list[str], path: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    for state in path:
        if state.startswith("I") or state.startswith("M"):
            result.append(ref[index])
            index += 1
        else:
            result.append("-")
    return result


def result_test():
    """
    Chạy các mẫu thử
    """
    global hmm
    print("Đang dóng hàng.")

    with open("Data/Out/result.txt", "w") as f_out:
        s_out: list[str] = ["".join(aligment_table[0])]
        for test_line in test_case:
            path = hmm.optimal_hidden_path(test_line)
            s_out.append("".join(align_from_hidden_path(test_line, path)))

        f_out.write("\n".join(s_out))

    print("Đã xong phần dóng hàng.")


def result_accuracy_test():
    """
    Kiểm tra độ chính xác của việc dóng hàng.
    """
    print("Đang kiểm tra độ chính xác dóng hàng.")

    with open("Data/Out/result.txt") as f_my_result, open("Data/In/data_checking.txt") as f_your_result, open("Data/Out/result_checking.txt", "w") as f_result_check:
        f_my_result.readline()
        distance_scores = []
        for my_line, your_line in zip(f_my_result, f_your_result):
            my_line = my_line.strip()
            your_line = your_line.strip()
            distance_scores.append(
                eval(my_line, your_line)/max(len(my_line), len(your_line)))
            f_result_check.write(f"{my_line}\n{your_line}\n\n")
        f_result_check.write(f"Avg score:\t{str(sum(distance_scores)/len(distance_scores))}")

    print("Đã xong phần kiểm tra độ chính xác dóng hàng.")


def speed_gen_test():
    """
    Kiểm tra thời gian dựng mô hình.
    """
    print("Đang kiểm tra thời gian dựng mô hình.")

    with open("Data/Out/hmm_gen_analytic.txt", "w") as f_out:
        for l in range(100, len(aligment_table), 100):
            t1 = time()
            _ = ProfileHMM(amino_acid, aligment_table[:l], pseu=0.01)
            t2 = time()
            f_out.write(f"{l}\t{t2-t1}\n")

    print("Đã xong phần kiểm tra thời gian dựng mô hình.")


def speed_exe_test():
    """
    Kiểm tra thời gian chạy thuật toán.
    """
    global hmm
    print("Đang kiểm tra thời gian chạy thuật toán.")

    test_line = test_case[0]
    for test_c in test_case:
        if (len(test_c) > len(test_line)):
            test_line = test_c

    with open("Data/Out/hmm_exe_analytic.txt", "w") as f_out:
        for i in range(50, len(test_line) + 9, 10):
            t1 = time()
            for __ in range(10):
                _ = hmm.optimal_hidden_path(test_line[:i])
            t2 = time()
            f_out.write(f"{i}\t{(t2-t1)/10}\n")

    print("Đã xong phần kiểm tra thời gian chạy thuật toán.")


def main():
    global aligment_table
    global test_case
    global hmm

    # Training data
    with open("Data/In/data_training.txt") as f_training:
        aligment_table = [list(line.strip()) for line in f_training]

    # Test data
    with open("Data/In/data_executing.txt") as f_test:
        test_case = [list(line.strip()) for line in f_test]

    # Test
    # speed_gen_test()
    t1 = time()
    hmm = ProfileHMM(amino_acid, aligment_table, pseu=0.01)
    t2 = time()
    print(f"Thoi gian dung mo hinh: {t2-t1}\n")
    # result_accuracy_test()
    # speed_exe_test()


main()
