import matplotlib.pyplot as plt
from utils import Dataset


class DataVisualizer(Dataset):
    def __init__(self, dataset):
        Dataset.__init__(self, dataset)
        pass

    def plot_single_state(self, ax, emv_state, ov_state):
        
        pass


if __name__ == "__main__":
    dv = DataVisualizer("emv_case1")
    print(dv.get_exact_trajectories())
