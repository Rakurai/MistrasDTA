import os.path as osp
import glob
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dtaDir",
        action="store",
        default=osp.join(osp.dirname(osp.abspath(__file__)), 'dta'),
        help="The directory with dta files."
    )
    parser.addoption(
        "--refDir",
        action="store",
        default=osp.join(osp.dirname(osp.abspath(__file__)), 'reference'),
        help="The directory with reference solutions."
    )


@pytest.fixture
def dta_dir(request):
    """Configurable root directory for .DTA test data."""
    return request.config.getoption('--dtaDir')


@pytest.fixture
def ref_dir(request):
    """Configurable root directory for reference solutions."""
    return request.config.getoption('--refDir')


@pytest.fixture
def dta_file(dta_dir, dta_stem):
    """Full path to the .DTA file under test."""
    return osp.join(dta_dir, f"{dta_stem}.DTA")


@pytest.fixture
def ref_file(ref_dir, dta_stem):
    """Full path to the reference .npz file for the current stem."""
    return osp.join(ref_dir, f"{dta_stem}.npz")


@pytest.fixture
def cont_files(dta_dir):
    """Sorted list of continuation files (filenames containing '__')."""
    return sorted(glob.glob(osp.join(dta_dir, '*__*.DTA')))


def pytest_generate_tests(metafunc):
    """Parametrize tests that request ``dta_stem`` over every standalone
    .DTA file discovered in ``--dtaDir`` (continuation files excluded)."""
    if "dta_stem" not in metafunc.fixturenames:
        return
    dta_dir = metafunc.config.getoption('--dtaDir')
    stems = sorted(
        osp.splitext(osp.basename(f))[0]
        for f in glob.glob(osp.join(dta_dir, '*.DTA'))
        if '__' not in osp.basename(f)
    )
    metafunc.parametrize("dta_stem", stems)
