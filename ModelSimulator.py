import numpy as np
from collections import defaultdict
from utils import Dataset 
from DataVisualizer import plot_multiple_states


class ModelSimulator(Dataset):
    """
    Simulate the model in a greedy fashion
    """
    def __init__(self, dataset):
        super().__init__(dataset)
        self.target_lane = self.get_target_lane()

    def get_target_lane(self):
        lane2ovs = defaultdict(int)
        for (_, l, _) in self.ov_init_state.values():
            lane2ovs[l] += 1
        target_lane = min(lane2ovs, key=lambda l:lane2ovs[l])
        return target_lane

    def perform_state_transition(self, cur_emv_state, cur_ov_state):
        next_emv_state = {m: list(state) for m, state in cur_emv_state.items()}
        next_ov_state = {n: list(state) for n, state in cur_ov_state.items()}

        for m, (i, l, v) in cur_emv_state.items():
            # if m not align, change lane
            if l > self.target_lane:
                next_emv_state[m][1] -= 1
            elif l < self.target_lane:
                next_emv_state[m][1] += 1
            
            # maintain speed and proceed
            next_emv_state[m][0] += v
        
        for n, (i, l, v) in cur_ov_state.items():
            # maintain speed and proceed
            next_ov_state[n][0] += v

        return next_emv_state, next_ov_state

    def run_model_simulation(self, time_horizon):
        cur_emv_state = self.emv_init_state
        cur_ov_state = self.ov_init_state
        lst_of_states = []

        for cur_time in range(time_horizon):
            lst_of_states.append((cur_time, cur_emv_state, cur_ov_state))
            cur_emv_state, cur_ov_state = self.perform_state_transition(cur_emv_state, cur_ov_state)

        plot_multiple_states(self.dataset, lst_of_states)


if __name__ == '__main__':
    ms = ModelSimulator("emv_case1")
    ms.run_model_simulation(5)

    









