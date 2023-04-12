from collections import defaultdict
from itertools import groupby
import json
import numpy as np


class Dataset(object):
    """
    Config Parameters:
        EMV_initial_state: (i, l, v) for each EMV @ t=0
        OV_initial_state: (i, l, v) for each OV @ t=0
        T(s, 15), I(# cells in a lane), L(# lanes), M(# EMV), N(# OV), V(cell/s)
        A(m,len), B(m,wid), a+/a-(cell/s^2)
        weight: [1, 1, 1]
    """
    def __init__(self, dataset):
        self.dataset = dataset
        with open("Datasets/" + dataset + "/%s.json"%dataset, 'r') as outfile:
            self.config = json.load(outfile)

        self.emv_init_state = {i+1: tuple(state) for i, state in enumerate(self.config["EMV_initial_state"])}
        self.ov_init_state = {i+1: tuple(state) for i, state in enumerate(self.config["OV_initial_state"])}

    def __repr__(self):
        str_res = ""
        inp_attribs = list(self.config.keys())[:13]
        for attrib, metadata in self.config.items():
            if attrib in inp_attribs:
                str_res += '\n' + attrib + ": " + str(metadata) + '\n'
                if type(metadata) == list:
                    str_res += "List length = %d \n"%len(metadata)
        return str_res
    
    def get_exact_trajectories(self):
        """
        Return (Gurobi results):
            emv_exact_traj: t:m:(i,l,v), nested dict, states of each emv from t=1 to t=T 
            ov_exact_traj: t:n:(i,l,v), nested dict, states of each ov from t=1 to t=T 
        """

        ov_traj_dist = np.reshape(self.config["OV_traj_dist"], (-1, self.config["N"]))
        ov_traj_lane = np.reshape(self.config["OV_traj_lane"], (-1, self.config["N"]))
        ov_traj_speed = np.reshape(self.config["OV_traj_speed"], (-1, self.config["N"]))
        emv_traj_lane = np.reshape(self.config["EMV_traj_lane"], (-1, self.config["M"]))
        
        ov_exact_traj, emv_exact_traj = {0: self.ov_init_state}, {0: self.emv_init_state}
        for t in range(self.config["T"]):
            ov_exact_traj[t+1] = {n+1: (ov_traj_dist[t][n], ov_traj_lane[t][n], ov_traj_speed[t][n]) for n in range(self.config["N"])}
            emv_exact_traj[t+1] = {}
            for m in range(self.config["M"]):
                dist = self.emv_init_state[m+1][0] + (t+1) * self.config["V"]
                emv_exact_traj[t+1][m+1] = (dist, emv_traj_lane[t][m], self.config["V"])
        
        return emv_exact_traj, ov_exact_traj

    def get_exact_changes(self, emv_exact_traj, ov_exact_traj):
        """
        Parameters:
            emv_exact_traj, ov_exact_traj: see Dataset.get_exact_trajectories()
        Return:
            sum_of_changes: sum of the length of the following dicts
            emv_change: {"lane_change":[(t, m, delta)], "speed_change":[(t, m, delta)]}
            ov_change: {"lane_change":[(t, n, delta)], "speed_change":[(t, n, delta)]}
        """
        emv_change, ov_change = defaultdict(list), defaultdict(list)
        emv_cur_state = {m: list(state) for m, state in self.emv_init_state.items()}
        ov_cur_state = {n: list(state) for n, state in self.ov_init_state.items()}

        sum_of_changes = 0
        for t in range(1, self.config["T"]+1):
            for m in range(1, self.config["M"]+1):
                if emv_exact_traj[t][m][1] != emv_cur_state[m][1]:
                    emv_change["lane_change"].append((t-1, m, emv_exact_traj[t][m][1]-emv_cur_state[m][1]))
                    emv_cur_state[m][1] = emv_exact_traj[t][m][1]
                    sum_of_changes += 1
                
                if emv_exact_traj[t][m][2] != emv_cur_state[m][2]:
                    emv_change["speed_change"].append((t-1, m, emv_exact_traj[t][m][2]-emv_cur_state[m][2]))
                    emv_cur_state[m][2] = emv_exact_traj[t][m][2]
                    sum_of_changes += 1
            
            for n in range(1, self.config["N"]+1):
                if ov_exact_traj[t][n][1] != ov_cur_state[n][1]:
                    ov_change["lane_change"].append((t-1, n, ov_exact_traj[t][n][1]-ov_cur_state[n][1]))
                    ov_cur_state[n][1] = ov_exact_traj[t][n][1]
                    sum_of_changes += 1
                
                if ov_exact_traj[t][n][2] != ov_cur_state[n][2]:
                    ov_change["speed_change"].append((t-1, n, ov_exact_traj[t][n][2]-ov_cur_state[n][2]))
                    ov_cur_state[n][2] = ov_exact_traj[t][n][2]
                    sum_of_changes += 1
        
        return sum_of_changes, emv_change, ov_change


class ConflictHelper(Dataset):
    """
    Resolving conflicts caused among EMVs and OVs
    """
    def __init__(self, dataset):
        super().__init__(dataset)

    def prepare_state_with_conflict(self, emv_state, ov_state):
        """
        Parameters:
            emv_state: {m: (i, l, v)}
            ov_state: {n: (i, l, v)}
        Return:
            color_mat: 2-d array, 0-empty-white, 1-ov-black, 2-emv-red, 3-cfl-orange
            text_lst: [(x, y, text)]
                Note: EMVs are renamed as alphabetic letters
        """
        color_mat = np.zeros((self.config["I"], self.config["L"]))
        loc2cars = defaultdict(list)
        
        for m, (i, l, _) in emv_state.items():
            m_str = chr(96 + m) 
            loc2cars[i, l].append(m_str)
            if color_mat[-i, -l] == 0:
                color_mat[-i, -l] = 2
            elif color_mat[-i, -l] == 2:
                color_mat[-i, -l] = 3

        for n, (i, l, _) in ov_state.items():
            n_str = str(n)
            loc2cars[i, l].append(n_str)
            if color_mat[-i, -l] == 0:
                color_mat[-i, -l] = 1
            elif color_mat[-i, -l] in [1, 2]:
                color_mat[-i, -l] = 3

        text_lst = [(i, l, "/".join(cars)) for (i, l), cars in loc2cars.items()]
        return color_mat, text_lst
