from HMM import *
from math import log


class _AligmentX:
    aligment_table: list[list[str]] = []
    aligment_choice: list[bool] = []

    def __init__(self, aligment_table: list[list[str]], theta: float = 1) -> None:
        self.aligment_table = aligment_table
        self.aligment_x = []
        len_vertical: int = len(aligment_table)

        for i in range(len_vertical):
            self.aligment_x.append([])

        len_horizontal: int = min(len(s) for s in aligment_table)
        self.aligment_choice = [False] * len_horizontal
        for i in range(len_horizontal):
            _count: int = 0
            for j in range(len_vertical):
                _count += aligment_table[j][i] == "-"

            if _count/len_vertical < theta:
                self.aligment_choice[i] = True

        # Khu tao bang chu cai
        self.alphabet = sorted(list(set(
            character for line in aligment_table for character in line if character != "-")))


class _Profile:
    aligment_x: _AligmentX
    profile: list[list[float]] = []

    def __init__(self, aligment_x: _AligmentX) -> None:
        self.aligment_x = aligment_x
        self.init_profile()

    def init_profile(self):
        self.profile.clear()
        for i in range(len(self.aligment_x.alphabet)):
            self.profile.append([0]*len(self.aligment_x.aligment_x[0]))

        for j in range(len(self.aligment_x.aligment_x[0])):
            column_j = [row[j]
                        for row in self.aligment_x.aligment_x if row[j] != "-"]
            for i in range(len(self.aligment_x.alphabet)):
                character = self.aligment_x.alphabet[i]
                self.profile[i][j] = column_j.count(
                    character) / len(column_j)


class ProfileHMM(HMM):
    aligment_x: _AligmentX

    def __init__(self, alphabet: list[str], aligment_table: list[list[str]], theta: float = 1, pseu: float = 0) -> None:
        self.aligment_x = _AligmentX(aligment_table, theta)

        _hidden_states: list[str] = ["S", "I0"]
        for i in range(self.aligment_x.aligment_choice.count(True)):
            _hidden_states.append("M{0}".format(i+1))
            _hidden_states.append("D{0}".format(i+1))
            _hidden_states.append("I{0}".format(i+1))
        _hidden_states.append("E")

        _transition: list[list[float]] = [
            [0.0] * len(_hidden_states) for _ in range(len(_hidden_states))]
        _emission: list[list[float]] = [
            [0.0] * len(alphabet) for _ in range(len(_hidden_states))]

        super().__init__(alphabet, _hidden_states, _transition, _emission)

        self.get_tran_and_emit()
        self.add_pseudocounts(pseu)

    def __str__(self) -> str:
        states: str = "\t\t".join(self.hidden_states)
        trans: str = ""
        alphabets: str = "\t\t".join(self.alphabet)
        emits: str = ""
        for i in range(len(self.transition)):
            trans += "{0}\t{1}\n".format(self.hidden_states[i], "\t".join(
                f"{tran:.3f}" for tran in self.transition[i]))
            emits += "{0}\t{1}\n".format(self.hidden_states[i], "\t".join(
                f"{emit:.3f}" for emit in self.emission[i]))

        return f"\t{states}\n{trans}--------\n\t{alphabets}\n{emits}"

    def hidden_path_from_align(self, x: list[str]) -> list[str]:
        kq: list[str] = []
        index: int = 0
        for i in range(len(x)):
            is_in_aligment_x: bool = self.aligment_x.aligment_choice[i]
            is_character: bool = x[i] != "-"
            if is_in_aligment_x and is_character:
                index += 1
                kq.append(f"M{index}")
            elif (not is_in_aligment_x) and is_character:
                kq.append(f"I{index}")
            elif is_in_aligment_x and (not is_character):
                index += 1
                kq.append(f"D{index}")
            else:
                kq.append("-")
        return kq

    def get_tran_and_emit(self) -> None:
        '''Tính bảng xác suất chuyển trạng thái và xác suất sinh chữ từ aligment đã có sẵn (chỉ dùng trong `__init__`)'''
        for align_row in self.aligment_x.aligment_table:
            path_row: list[str] = self.hidden_path_from_align(align_row)
            for _character, _state in zip(align_row, path_row):
                if _state == "-" or _state.startswith("D"):
                    continue
                c_index = self.find_character_index(_character)
                s_index = self.find_state_index(_state)
                self.emission[s_index][c_index] += 1

            state_from_index: int = 0
            for _state in path_row:
                if _state != "-":
                    state_to_index: int = self.find_state_index(_state)
                    self.transition[state_from_index][state_to_index] += 1
                    state_from_index = state_to_index
            self.transition[state_from_index][-1] += 1

        self.normalize_matrix()

    def normalize_matrix(self) -> None:
        # Chuan hoa emission
        for emis_row in self.emission:
            total_count: float = sum(emis_row)
            if total_count > 0:
                emis_row[:] = list(emis/total_count for emis in emis_row)
        # Chuan hoa transition
        for tran_row in self.transition:
            total_count: float = sum(tran_row)
            if total_count > 0:
                tran_row[:] = list(tran/total_count for tran in tran_row)

    def add_pseudocounts(self, pseu: float = 0) -> None:
        # pseudocounts for transitions
        for j in range(2):
            for k in range(1, 4):
                self.transition[j][k] += pseu
        for i in range(1, self.aligment_x.aligment_choice.count(True)):
            for j in range(3*i-1, 3*i+2):
                for k in range(3*i+1, 3*i+4):
                    self.transition[j][k] += pseu
        for j in range(-2, -5, -1):
            for k in range(-1, -3, -1):
                self.transition[j][k] += pseu

        # pseudocounts for emissions
        for i in range(len(self.transition)-1):
            if i % 3 == 0:
                continue

            emit_row = self.emission[i]
            emit_row[:] = [character + pseu for character in emit_row]

        self.normalize_matrix()

    def optimal_hidden_path(self, x: list[str]) -> list[str]:
        dynamic_path: list[list[float]] = []
        trace: list[list[int]] = []  # 0 <=> M, 1 <=> D, 2 <=> I

        # Khu khởi tạo giá trị
        for _ in range(len(self.hidden_states) - 2):
            dynamic_path.append([-float('inf')] * (len(x) + 1))
            trace.append([-1] * (len(x)+1))

        # Tính bảng theo thứ tự tô-pô (cột đầu, gồm các D)
        dynamic_path[0][0] = 0
        dynamic_path[2][0] = log(self.transition[0][3])
        trace[2][0] = 2
        for i in range(5, len(dynamic_path), 3):
            dynamic_path[i][0] = dynamic_path[i-3][0] + \
                log(self.transition[i-2][i+1])
            trace[i][0] = 1

        # Tính bảng theo thứ tự tô-pô (những cột còn lại)
        for i in range(1, len(x)+1):
            index_character = self.find_character_index(x[i-1])

            # Hàng I0
            dynamic_path[0][i] = dynamic_path[0][i-1] + \
                log(self.transition[0 if i == 1 else 1][1]) + \
                log(self.emission[1][index_character])
            trace[0][i] = 2
            # Hàng M1
            dynamic_path[1][i] = dynamic_path[0][i-1] + \
                log(self.transition[0 if i == 1 else 1][2]) + \
                log(self.emission[2][index_character])
            trace[1][i] = 2
            # Hàng D1
            dynamic_path[2][i] = dynamic_path[0][i] + \
                log(self.transition[1][3])
            trace[2][i] = 2

            # Những hàng còn lại
            for j in range(3, len(dynamic_path)):
                compares: list[float] = []
                if j % 3 == 0:      # [j][i] = I
                    for k in range(j-2, j+1):
                        compares.append(
                            dynamic_path[k][i-1] + log(self.transition[k+1][j+1]) + log(self.emission[j+1][index_character]))
                elif j % 3 == 1:    # [j][i] = M
                    for k in range(j-3, j):
                        compares.append(
                            dynamic_path[k][i-1] + log(self.transition[k+1][j+1]) + log(self.emission[j+1][index_character]))
                else:               # [j][i] = D
                    for k in range(j-4, j-1):
                        compares.append(
                            dynamic_path[k][i] + log(self.transition[k+1][j+1]))
                dynamic_path[j][i] = max(compares)
                trace[j][i] = compares.index(dynamic_path[j][i])

        # Khu truy vết
        compares: list[float] = []
        compares.append(dynamic_path[-3][-1]+log(self.transition[-4][-1]))  # Mx -> E
        compares.append(dynamic_path[-2][-1]+log(self.transition[-3][-1]))  # Dx -> E
        compares.append(dynamic_path[-1][-1]+log(self.transition[-2][-1]))  # Ix -> E

        trace_int: list[int] = []
        pointer_x: int = len(x)
        pointer_y: int = len(dynamic_path) - 3 + compares.index(max(compares))
        while pointer_x or pointer_y:
            trace_int.append(pointer_y + 1)
            previous_step: int = trace[pointer_y][pointer_x]
            if pointer_y % 3 == 0:      # I
                pointer_y = pointer_y - 2 + previous_step
                pointer_x = pointer_x - 1
            elif pointer_y % 3 == 1:    # M
                pointer_y = pointer_y - 3 + previous_step
                pointer_x = pointer_x - 1
            else:                       # D
                pointer_y = pointer_y - 4 + previous_step
                pointer_x = pointer_x - 0

        return [self.hidden_states[s] for s in trace_int[::-1]]
