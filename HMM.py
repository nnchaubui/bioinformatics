class HMM:
    '''Hidden Markov Model'''
    alphabet: list[str]
    '''Bảng chữ cái'''
    hidden_states: list[str]
    '''Tập trạng thái ẩn'''
    transition: list[list[float]]
    emission: list[list[float]]

    def __init__(self, alphabet: list[str] = [], hidden_states: list[str] = [], transition: list[list[float]] = [], emission: list[list[float]] = []) -> None:
        self.alphabet = alphabet
        self.hidden_states = hidden_states
        self.transition = transition
        self.emission = emission

    def __str__(self) -> str:
        kq = ""
        kq += "\t".join(self.alphabet) + "\n"
        kq += "\t".join(self.hidden_states) + "\n\n"
        kq += "\n".join(["\t".join(str(p) for p in ps)
                        for ps in self.transition]) + "\n\n"
        kq += "\n".join(["\t".join(str(p) for p in ps)
                        for ps in self.emission])
        return kq

    def check_valid(self) -> None:  # TODO check valid
        '''Kiểm tra xem HMM này có hợp lệ không, với các tiêu chí sau (theo thứ tự):
          - Số chữ cái trong bảng chữ cái (|`Σ`|) > 0
          - Số trạng thái ẩn (|`States`|) > 0
          - `Transition` có kích cỡ đúng bằng |`States`| x |`States`|
          - Với mỗi hàng `l` trong `Transition`, tổng tất cả các phần tử trong hàng bằng 1
          - `Emission` có kích cỡ đúng bằng |`States`| x |`Σ`|
          - Với mỗi cột `b` trong `Emission`, tổng tất cả các phần tử trong hàng bằng 1
          '''
        pass

    def find_character_index(self, character: str) -> int:
        return self.alphabet.index(character)

    def find_state_index(self, state: str) -> int:
        return self.hidden_states.index(state)

    def pr_pi(self, pi: list[str]) -> float:
        '''Tính xác suất để HMM này tạo được chuỗi ẩn `π` cho trước'''
        if not len(pi):
            return 0

        total_probability: float = 1/len(self.hidden_states)

        state_to: int = self.find_state_index(pi[0])
        for i in range(1, len(pi)):
            state_from = state_to
            state_to = self.find_state_index(pi[i])
            total_probability *= self.transition[state_from][state_to]
        return total_probability
    
    def pr_x(self, x: list[str]) -> float:
        '''Tính xác suất để HMM này sinh ra được chuỗi `x` cho trước'''
        dynamic_path: list[list[float]] = []

        # Khu khởi tạo giá trị
        for _ in range(len(self.hidden_states)):
            dynamic_path.append([0] * len(x))

        index_character = self.find_character_index(x[0])
        for i in range(len(self.hidden_states)):
            dynamic_path[i][0] = self.emission[i][index_character] / len(self.hidden_states)

        for i in range(1, len(x)):
            index_character = self.find_character_index(x[i])
            for j in range(len(self.hidden_states)):
                sumups: list[float] = []
                for k in range(len(self.hidden_states)):
                    sumups.append(
                        dynamic_path[k][i-1] * self.transition[k][j] * self.emission[j][index_character])

                dynamic_path[j][i] = sum(sumups)

        sumups = [sumup[-1] for sumup in dynamic_path]
        return sum(sumups)

    def pr_x_know_pi(self, x: list[str], pi: list[str]) -> float:
        '''Tính xác suất để HMM này sinh ra được xâu `x` từ chuỗi ẩn `π` cho trước'''
        if len(x) != len(pi):
            return 0

        total_probability = 1
        for i in range(len(x)):
            state_index = self.find_state_index(pi[i])
            character_index = self.find_character_index(x[i])
            total_probability *= self.emission[state_index][character_index]
        return total_probability

    def pr_x_and_pi(self, x: list[str], pi: list[str]) -> float:
        '''Tính xác suất để HMM này sinh ra được xâu `x` đồng thời chuỗi ẩn là `π`'''
        if len(x) != len(pi):
            return 0

        character_index = self.find_character_index(x[0])
        state_index = self.find_state_index(pi[0])
        total_probability = 1/len(self.hidden_states) * \
            self.emission[state_index][character_index]

        for i in range(1, len(x)):
            previous_index = state_index
            state_index = self.find_state_index(pi[i])
            character_index = self.find_character_index(x[i])
            total_probability *= self.emission[state_index][character_index] * \
                self.transition[previous_index][state_index]

        return total_probability

    def optimal_hidden_path(self, x: list[str]) -> list[str]:
        dynamic_path: list[list[float]] = []
        trace: list[list[int]] = []

        # Khu khởi tạo giá trị
        for _ in range(len(self.hidden_states)):
            dynamic_path.append([0] * len(x))
            trace.append([-1] * len(x))

        index_character = self.find_character_index(x[0])
        for i in range(len(self.hidden_states)):
            dynamic_path[i][0] = self.emission[i][index_character]

        for i in range(1, len(x)):
            index_character = self.find_character_index(x[i])
            for j in range(len(self.hidden_states)):
                compares: list[float] = []
                for k in range(len(self.hidden_states)):
                    compares.append(
                        dynamic_path[k][i-1] * self.transition[k][j] * self.emission[j][index_character])

                dynamic_path[j][i] = max(compares)
                trace[j][i-1] = compares.index(dynamic_path[j][i])

        compares = [compare[-1] for compare in dynamic_path]
        trace_int: list[int] = []
        trace_int.append(compares.index(max(compares)))
        for inverse_i in range(len(x)-1)[::-1]:
            trace_int.append(trace[trace_int[-1]][inverse_i])

        return [self.hidden_states[s] for s in trace_int[::-1]]
