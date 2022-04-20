from orbit_calibration_2_fullsky_map.spline_fit_calibration import SplineFitter
import sys

if __name__ == "__main__":
    iterations = int(sys.argv[1])
    sf = SplineFitter(iterations)
    for i in range(0, 212, 10):
        sf._plot_fit_evolution(i)
