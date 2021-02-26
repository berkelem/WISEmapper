from orbit_calibration_2_fullsky_map.spline_fit_calibration import SplineFitter


if __name__ == "__main__":
    sf = SplineFitter(25)
    for i in range(0, 212, 10):
        sf._plot_fit_evolution(i)
