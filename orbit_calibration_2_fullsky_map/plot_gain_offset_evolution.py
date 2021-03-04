from orbit_calibration_2_fullsky_map.spline_fit_calibration import SplineFitter
import sys

if __name__ == "__main__":
    for i in range(sys.argv[1], sys.argv[2], sys.argv[3]):
        sf = SplineFitter(i)
        sf._plot_all_fitvals()