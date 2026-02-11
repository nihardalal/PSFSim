"""Test functions for diffusion."""


import numpy as np
import psfsim.mtf_diffusion as mtfd


def test_green():
    """Green's function test."""

    xd, yd = np.meshgrid(np.linspace(0.0, 10.0, 11), np.linspace(0.0, 10.0, 11))
    g = mtfd.diffusion_green(xd, yd, x2=5.0, y2=4.0)
    w = np.sum(g)
    x_ = np.sum(g * xd) / w
    y_ = np.sum(g * yd) / w
    varx = np.sum(g * (xd - x_) ** 2) / w
    vary = np.sum(g * (yd - y_) ** 2) / w
    print(w, x_, y_, varx, vary)
    assert w > 0.8
    assert w < 1.0
    assert np.abs(x_ - 5.0) < 0.3
    assert np.abs(y_ - 4.0) < 0.3
    assert varx > 5.0
    assert vary > 5.0
    assert varx < 6.0
    assert vary < 6.0

    g2 = mtfd.diffusion_prob(xd, yd, width=1.0, x2=5.0, y2=4.0)

    print(np.amax(g), np.amax(np.abs(g - g2)))
    assert np.amax(np.abs(g - g2)) / np.amax(g) < 0.03


def test_convolve():
    """Test for convolution with Green's function."""

    xi_ = np.linspace(20320.0, 20520.0, 81)
    yi_ = np.linspace(300.0, 500.0, 81)
    xi, yi = np.meshgrid(xi_, yi_)
    intensity = np.exp(-0.5 * ((xi - 20460.0) ** 2 + (yi - 412.0) ** 2) / 5.0**2) / (2.0 * np.pi * 5.0**2)
    i_conv1 = mtfd.intensity_to_image(
        intensity, 20420.0, 410.0, 20422.0, 408.0, 33, 2.5, reflect=False, tophat=False
    )
    i_conv2 = mtfd.intensity_to_image(
        intensity, 20420.0, 410.0, 20422.0, 408.0, 33, 2.5, reflect=True, tophat=False
    )

    assert np.abs(i_conv1[21, 15] / 0.0007185) < 0.01
    assert np.abs(i_conv1[21, 31] / 0.0007185 - 1.0) < 0.01
    assert np.abs(i_conv2[21, 15] / 0.0007185 - 1.0) < 0.01
    assert np.abs(i_conv2[21, 31] / 0.0007185 - 1.0) < 0.01

    # test transpose
    i_conv3 = mtfd.intensity_to_image(
        intensity.T, 410.0, 20420.0, 408.0, 20422.0, 33, 2.5, reflect=False, tophat=False
    )
    i_conv4 = mtfd.intensity_to_image(
        intensity.T, 410.0, 20420.0, 408.0, 20422.0, 33, 2.5, reflect=True, tophat=False
    )
    assert np.all(np.abs(i_conv1 - i_conv3.T) < 1.0e-6)
    assert np.all(np.abs(i_conv2 - i_conv4.T) < 1.0e-6)

    # fits.PrimaryHDU(np.stack((i_conv1, i_conv2))).writeto("~/testimage.fits", overwrite=True)

    # test tophat
    i_conv5 = mtfd.intensity_to_image(
        intensity, 20420.0, 410.0, 20422.0, 408.0, 33, 2.5, reflect=False, tophat=True
    )
    x0 = 28
    y0 = 20
    q = np.array([0.125, 0.25, 0.25, 0.25, 0.125])
    comparison = np.sum(np.outer(q, q) * i_conv1[y0 - 2 : y0 + 3, x0 - 2 : x0 + 3]) * 100.0

    print(i_conv5[y0, x0], comparison)
    assert np.abs(i_conv5[y0, x0] / comparison - 1.0) < 0.01
