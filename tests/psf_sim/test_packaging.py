import psf_sim


def test_version():
    """Check to see that we can get the package version"""
    assert psf_sim.__version__ is not None
