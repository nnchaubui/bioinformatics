from HMM import *


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
