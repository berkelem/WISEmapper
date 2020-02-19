import pickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

def main():
    filename = "/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/popt_w3.pkl"
    with open(filename, "rb") as f:
        gain, offset = pickle.load(f)

    unreliable_fits = [4, 7, 13, 25, 46, 66, 87, 438, 482, 484, 486, 516, 520, 522, 749, 759, 795, 825, 839, 869, 971, 993, 1006,
                       1031, 1050, 1091, 1096, 1110, 1123, 1156, 1186, 1202, 1232, 1294, 1303, 1315, 1317, 1324, 1333, 1335,
                       1345, 1347, 1354, 1355, 1371, 1373, 1375, 1377, 1379, 1398, 1399, 1401, 1403, 1405, 1407, 1409, 1411,
                       1413, 1414, 1415, 1444, 1460, 1474, 1504, 1534, 1575, 1587, 1607, 1617, 1619, 1635, 1637, 1647, 1649,
                       1657, 1667, 1677, 1679, 1688, 1696, 1707, 1709, 1711, 1735, 1737, 1739, 1741, 1748, 1756, 1759, 1763,
                       1765, 1767, 1771, 1773, 1779, 1781, 1783, 1788, 1789, 1793, 1795, 1797, 1803, 1805, 1806, 1808, 1809,
                       1811, 1813, 1815, 1817, 1819, 1823, 1825, 1827, 1833, 1839, 1843, 1849, 1855, 1857, 1863, 1869, 1871,
                       1875, 1878, 1883, 1885, 1887, 1895, 1901, 1903, 1905, 1913, 1915, 1931, 1932, 1933, 1935, 1936, 1938,
                       1943, 1944, 1945, 1948, 1952, 1955, 1956, 1959, 1960, 1962, 1963, 1964, 1970, 1974, 1975, 1979, 1983,
                       1984, 1989, 1990, 1991, 1992, 1994, 1995, 2000, 2003, 2004, 2007, 2011, 2014, 2015, 2016, 2020, 2022,
                       2023, 2024, 2029, 2030, 2031, 2034, 2035, 2039, 2043, 2044, 2046, 2050, 2051, 2052, 2054, 2057, 2058,
                       2062, 2069, 2073, 2077, 2163, 2167, 2175, 2183, 2187, 2191, 2195, 2203, 2207, 2214, 2219, 2223, 2227,
                       2228, 2235, 2239, 2240, 2242, 2247, 2251, 2255, 2259, 2260, 2262, 2267, 2270, 2272, 2279, 2283, 2287,
                       2290, 2295, 2296, 2299, 2300, 2302, 2304, 2307, 2308, 2311, 2315, 2319, 2323, 2335, 2336, 2339, 2361,
                       2366, 2381, 2383, 2391, 2393, 2396]
    orbits = np.arange(len(gain)) + 1

    bad_fit_orbits = orbits[unreliable_fits]
    bad_fit_gains = gain[unreliable_fits]
    bad_fit_offset = offset[unreliable_fits]

    mask = np.zeros_like(gain, dtype=bool)
    mask[unreliable_fits] = 1.0
    mask[gain == 0.0] = 1.0
    gain = ma.array(gain, mask=mask)
    offset = ma.array(offset, mask=mask)

    i_vals = np.arange(0, len(gain)-1, 2)

    merged_gain = np.array([ma.mean([gain[i], gain[i+1]]) for i in i_vals])
    merged_offset = np.array([ma.mean([offset[i], offset[i+1]]) for i in i_vals])



    orbits[unreliable_fits] = 0
    gain[unreliable_fits] = 0
    offset[unreliable_fits] = 0


    l1, l2 = plt.plot(orbits[gain != 0.0], gain[gain != 0.0], "r.", bad_fit_orbits, bad_fit_gains, "b.")
    plt.xlabel("Scan number")
    plt.ylabel("Gain")
    plt.legend((l1, l2), ("Good fits", "Bad fits"))
    plt.title("Gain calibration factor")
    plt.savefig("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/gain_w3.png")
    plt.close()

    x = orbits[gain != 0.0]
    x_even = x[x % 2 == 0.0]
    x_odd = x[x % 2 == 1.0]

    y1 = gain[gain != 0.0]
    y1_even = y1[x % 2 == 0.0]
    y1_odd = y1[x % 2 == 1.0]

    plt.plot(x_even, y1_even, "r.")
    plt.xlabel("Scan number")
    plt.ylabel("Gain")
    plt.title("Even scans")
    plt.savefig("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/gain_w3_even.png")
    plt.close()

    plt.plot(x_odd, y1_odd, "r.")
    plt.xlabel("Scan number")
    plt.ylabel("Gain")
    plt.title("Odd scans")
    plt.savefig("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/gain_w3_odd.png")
    plt.close()

    plt.plot(i_vals, merged_gain, "r.")
    plt.xlabel("Scan number")
    plt.ylabel("Gain")
    plt.title("Merged scans")
    plt.savefig("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/gain_w3_merged.png")
    plt.close()

    y2 = offset[offset != 0.0]
    y2_even = y2[x % 2 == 0.0]
    y2_odd = y2[x % 2 == 1.0]

    plt.plot(x_even, y2_even, "b.")
    plt.xlabel("Scan number")
    plt.ylabel("Offset")
    plt.title("Even scans")
    plt.savefig("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/offset_w3_even.png")
    plt.close()

    plt.plot(x_odd, y2_odd, "b.")
    plt.xlabel("Scan number")
    plt.ylabel("Offset")
    plt.title("Odd scans")
    plt.savefig("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/offset_w3_odd.png")
    plt.close()

    plt.plot(i_vals, merged_offset, "b.")
    plt.xlabel("Scan number")
    plt.ylabel("Offset")
    plt.title("Merged scans")
    plt.savefig(
        "/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/offset_w3_merged.png")
    plt.close()

    l1, l2 = plt.plot(orbits[offset != 0.0], offset[offset != 0.0], "b.", bad_fit_orbits, bad_fit_offset, "r.")
    plt.xlabel("Scan number")
    plt.ylabel("Offset")
    plt.legend((l1, l2), ("Good fits", "Bad fits"))
    plt.title("Offset calibration factor")
    plt.savefig("/Users/Laptop-23950/projects/wisemapping/data/output_maps/orbit_analysis/calibration/offset_w3.png")
    plt.close()

if __name__ == "__main__":
    main()