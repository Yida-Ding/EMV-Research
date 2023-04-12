import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from utils import Dataset, ConflictHelper

class DataVisualizer(Dataset):
    """
    Visualize each frame from t=0 to t=T
    """
    def __init__(self, dataset, max_disp=20):
        super().__init__(dataset)
        self.max_disp = max_disp
        self.cfh = ConflictHelper(dataset)

    def plot_single_state(self, ax, cur_time, emv_state, ov_state):
        """
        Parameters:
            cur_time: t
            emv_state: {m:(i,l,v)}
            ov_state: {n:(i,l,v)}
        Visualize:
            a single state of emvs and ovs at a specified time step
        """
        # generate color_mat considering conflicts and text annotations
        color_mat, text_lst = self.cfh.prepare_state_with_conflict(emv_state, ov_state)

        # display the last max_disp # of rows
        color_mat = color_mat[- self.max_disp:]

        # make a 3d numpy array that has a color channel dimension   
        color_map = {0: np.array([255, 255, 255]), 1: np.array([0, 0, 0]), 2: np.array([255, 0, 0]), 3: np.array([255, 165, 0])}
        data_3d = np.ndarray(shape=(color_mat.shape[0], color_mat.shape[1], 3), dtype=int)
        for i in range(0, color_mat.shape[0]):
            for j in range(0, color_mat.shape[1]):
                data_3d[i][j] = color_map[color_mat[i][j]]

        ax.imshow(data_3d, aspect=self.config["B"] / self.config["A"])
        text_param = {"color":'white', "horizontalalignment":'center', "verticalalignment":'center', "fontsize":10}
        for i, l, text in text_lst:
            if i <= self.max_disp:
                ax.text(color_mat.shape[1]-l, color_mat.shape[0]-i, text, **text_param)

        ax.set_title("t=%d"%cur_time, fontsize=18)
        ax.set_xticks(np.arange(0.5, color_mat.shape[1]+0.5, step=1))
        ax.set_yticks(np.arange(0.5, color_mat.shape[0]+0.5, step=1))
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        ax.grid()

def plot_multiple_states(dataset, lst_of_states):
    """
    Parameters:
        dataset: name of the Dataset instance
        lst_of_states: [(cur_time, emv_state, ov_state)]
    """
    num_states = len(lst_of_states)
    fig, axes = plt.subplots(1, num_states, figsize=(2*num_states, 8))
    dv = DataVisualizer(dataset)
    for idx, (cur_time, emv_state, ov_state) in enumerate(lst_of_states):
        dv.plot_single_state(axes[idx], cur_time, emv_state, ov_state)

    plt.tight_layout()
    plt.show()
    
