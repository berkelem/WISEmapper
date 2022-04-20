import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np


class SplinePlot:

    def __init__(self):
        self.output_path = os.getcwd()
        self.gain_spline_file_days = os.path.join(self.output_path, "gain_spline_days.pkl")
        self.offset_spline_file_days = os.path.join(self.output_path, "offset_spline_days.pkl")
        self.gain_spline_file_orbits_offg = os.path.join(self.output_path, "gain_spline_orbits_offg.pkl")
        self.offset_spline_file_orbits_offg = os.path.join(self.output_path, "offset_spline_orbits_offg.pkl")
        self.gain_spline_file_orbits_ong = os.path.join(self.output_path, "gain_spline_orbits_ong.pkl")
        self.offset_spline_file_orbits_ong = os.path.join(self.output_path, "offset_spline_orbits_ong.pkl")
        self.times = np.arange(55200, 55420, 0.1)


    def load_splines(self):
        """Load gain spline and offset spline from '*.pkl' files saved in a file"""
        with open(self.gain_spline_file_days, "rb") as g1:
            self.gain_spline_days = pickle.load(g1)

        with open(self.offset_spline_file_days, "rb") as g2:
            self.offset_spline_days = pickle.load(g2)

        with open(self.gain_spline_file_orbits_offg, "rb") as g3:
            self.gain_spline_orbits_offg = pickle.load(g3)

        with open(self.offset_spline_file_orbits_offg, "rb") as g4:
            self.offset_spline_orbits_offg = pickle.load(g4)

        with open(self.gain_spline_file_orbits_ong, "rb") as g5:
            self.gain_spline_orbits_ong = pickle.load(g5)

        with open(self.offset_spline_file_orbits_ong, "rb") as g6:
            self.offset_spline_orbits_ong = pickle.load(g6)

    def plot_splines(self):
        str_month_dict = OrderedDict([(55197, "Jan"), (55228, "Feb"), (55256, "Mar"), (55287, "Apr"),
                                      (55317, "May"), (55348, "Jun"), (55378, "Jul"), (55409, "Aug")])

        min_time = min(self.times)
        max_time = max(self.times)
        month_start_times = list(str_month_dict.keys())

        start_month_ind = month_start_times.index(min(month_start_times, key=lambda x: abs(x - min_time)))
        start_month_ind = start_month_ind if month_start_times[start_month_ind] < min_time else start_month_ind - 1

        end_month_ind = month_start_times.index(min(month_start_times, key=lambda x: abs(x - max_time)))
        end_month_ind = end_month_ind if month_start_times[end_month_ind] > max_time else end_month_ind + 1

        x_ticks = month_start_times[start_month_ind:end_month_ind + 1]

        fig, ax = plt.subplots()
        ax.plot(self.times, self.gain_spline_days(self.times), 'ko', alpha=0.2, ms=3, label="Days")
        ax.plot(self.times, self.gain_spline_orbits_offg(self.times), 'ro', ms=3, label="Off-galaxy Orbits")
        ax.plot(self.times, self.gain_spline_orbits_ong(self.times), 'bo', ms=3, label="On-galaxy Orbits")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str_month_dict[x] for x in x_ticks], rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.xlabel("Orbit median timestamp")
        plt.ylabel("Gain spline")
        plt.legend()
        plt.savefig(os.path.join(self.output_path, "gain_splines.png"))
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.times, self.offset_spline_days(self.times), 'ko', alpha=0.2, ms=3, label="Days")
        ax.plot(self.times, self.offset_spline_orbits_offg(self.times), 'ro', ms=3, label="Off-galaxy Orbits")
        ax.plot(self.times, self.offset_spline_orbits_ong(self.times), 'bo', ms=3, label="On-galaxy Orbits")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str_month_dict[x] for x in x_ticks], rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.xlabel("Orbit median timestamp")
        plt.ylabel("Offset spline")
        plt.legend()
        plt.savefig(os.path.join(self.output_path, "offset_splines.png"))
        plt.close()

if __name__ == "__main__":
    sp = SplinePlot()
    sp.load_splines()
    sp.plot_splines()