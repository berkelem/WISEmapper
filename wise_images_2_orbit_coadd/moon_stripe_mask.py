import healpy as hp
import numpy as np
from wise_images_2_orbit_coadd.file_handler import HealpixMap


class MoonStripeMask(HealpixMap):
    def __init__(self, filename):
        super().__init__(filename)
        self.set_resolution(nside=256)

    def set_stripe_locations(self):
        phi1 = np.array(
            [
                [i] * 1000
                for i in np.arange(
                    0 + 9 * np.pi / 144.0,
                    (19.0 / 20.0) * np.pi + 9 * np.pi / 144.0,
                    np.pi / 6.0 - 1.8 * np.pi / 360.0,
                )
            ]
        ).T.flatten()  # from 9/144 pi to (19/20 pi + 9/144 pi) in steps of (pi/6 - 1.8/360 pi)

        theta1 = np.array(
            [
                [i] * 6
                for i in np.arange(
                    np.pi / 6.0, 5 * np.pi / 6.0, (2 / 3.0) * np.pi / 999.0
                )
            ]
        ).flatten()

        phi2 = np.array(
            [
                [i] * 1000
                for i in np.arange(
                    np.pi + 189 * np.pi / 720.0,
                    2 * np.pi,
                    (5 / 6.0) * np.pi / 5.0 - 1.8 * np.pi / 360.0,
                )
            ]
        ).T.flatten()
        theta2 = np.array(
            [
                [i] * 5
                for i in np.arange(
                    np.pi / 6.0, 5 * np.pi / 6.0, (2 / 3.0) * np.pi / 999.0
                )
            ]
        ).flatten()

        phi3 = np.array(
            [
                [i] * 1000
                for i in np.arange(
                    np.pi + (6 / 360.0) * np.pi,
                    (5 / 4.0) * np.pi,
                    (1 / 3.0) * np.pi / 2.0 - 1.9 * np.pi / 360.0,
                )
            ]
        ).T.flatten()
        theta3 = np.array(
            [
                [i] * 2
                for i in np.arange(
                    np.pi / 6.0, 5 * np.pi / 6.0, (2 / 3.0) * np.pi / 999.0
                )
            ]
        ).flatten()

        px_nums1 = hp.ang2pix(self.nside, theta1, phi1)
        px_nums2 = hp.ang2pix(self.nside, theta2, phi2)
        px_nums3 = hp.ang2pix(self.nside, theta3, phi3)

        return np.concatenate((px_nums1, px_nums2, px_nums3))

    def set_stripe_thickness(self, n, stripe_pixels):
        it = 0
        while it < n:
            nbs = hp.get_all_neighbours(self.nside, stripe_pixels)
            nbs = nbs.flatten()
            stripe_pixels = np.concatenate((stripe_pixels, nbs))
            it += 1
        return stripe_pixels

    def fill_mask(self, stripe_pixels):
        self.mapdata[stripe_pixels] = 1.0


def create_mask():
    wise_data = np.zeros(hp.nside2npix(256))
    rot_map = rotate_map(wise_data, rot=["E", "G"])
    hp.fitsfunc.write_map("rotated_map.fits", rot_map, coord="E", overwrite=True)
    npix = len(rot_map)
    nside = hp.npix2nside(npix)

    """Create moon stripe mask"""

    phi = np.array(
        [
            [i] * 1000
            for i in np.arange(
                0 + 9 * np.pi / 144.0,
                (19.0 / 20.0) * np.pi + 9 * np.pi / 144.0,
                np.pi / 6.0 - 1.8 * np.pi / 360.0,
            )
        ]
    ).T.flatten()  # ange(0, 2*np.pi, 2*np.pi/100.)
    # print 'phi', phi, phi.shape
    theta = np.array(
        [
            [i] * 6
            for i in np.arange(np.pi / 6.0, 5 * np.pi / 6.0, (2 / 3.0) * np.pi / 999.0)
        ]
    ).flatten()  # (np.repeat([3*np.pi/4., np.pi/2., np.pi/4.], 100)#0, np.pi, np.pi/100.)#ones_like(phi)*np.pi/2.
    # print 'theta', theta, theta.shape
    px_nums = hp.ang2pix(nside, theta, phi)

    phi2 = np.array(
        [
            [i] * 1000
            for i in np.arange(
                np.pi + 189 * np.pi / 720.0,
                2 * np.pi,
                (5 / 6.0) * np.pi / 5.0 - 1.8 * np.pi / 360.0,
            )
        ]
    ).T.flatten()
    theta2 = np.array(
        [
            [i] * 5
            for i in np.arange(np.pi / 6.0, 5 * np.pi / 6.0, (2 / 3.0) * np.pi / 999.0)
        ]
    ).flatten()
    px_nums2 = hp.ang2pix(nside, theta2, phi2)
    # print 'phi2', phi2.shape
    # print 'theta2', theta2.shape

    phi3 = np.array(
        [
            [i] * 1000
            for i in np.arange(
                np.pi + (6 / 360.0) * np.pi,
                (5 / 4.0) * np.pi,
                (1 / 3.0) * np.pi / 2.0 - 1.9 * np.pi / 360.0,
            )
        ]
    ).T.flatten()
    theta3 = np.array(
        [
            [i] * 2
            for i in np.arange(np.pi / 6.0, 5 * np.pi / 6.0, (2 / 3.0) * np.pi / 999.0)
        ]
    ).flatten()
    # print 'phi3', phi3.shape
    # print 'theta3', theta3.shape
    px_nums3 = hp.ang2pix(nside, theta3, phi3)

    stripe_mask = np.zeros_like(wise_data)
    stripe_mask[px_nums] = 1.0
    stripe_mask[px_nums2] = 1.0
    stripe_mask[px_nums3] = 1.0
    it = 0
    while it < 3:
        nbs = hp.get_all_neighbours(nside, np.arange(npix)[stripe_mask.astype(bool)])
        # print 'nbs shape', nbs.shape, nbs[0]
        nbs = nbs.flatten()
        stripe_mask[nbs] = 1.0
        it += 1
    stripe_mask_out = hp.ud_grade(np.copy(stripe_mask), 256)
    hp.fitsfunc.write_map(
        "stripe_mask.fits", stripe_mask_out, coord="E", overwrite=True
    )


if __name__ == "__main__":
    moon_stripe_mask = MoonStripeMask("moon_stripe_mask_E.fits")
    stripe_pixels = moon_stripe_mask.set_stripe_locations()
    stripe_pixels = moon_stripe_mask.set_stripe_thickness(6, stripe_pixels)
    moon_stripe_mask.fill_mask(stripe_pixels)
    moon_stripe_mask.save_map(coord="E")

    moon_stripe_mask.rotate_map(old_coord="E", new_coord="G")
    moon_stripe_mask.filename = "moon_stripe_mask_G_thick6.fits"
    moon_stripe_mask.save_map(coord="G")
