import os.path as osp
import glob


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


def pytest_generate_tests(metafunc):
    dta_dir = metafunc.config.getoption('--dtaDir')
    ref_dir = metafunc.config.getoption('--refDir')

    # Generate list of files to compare
    dta_files = glob.glob(dta_dir + '/**/*.DTA', recursive=True)
    ref_files = [
        osp.join(ref_dir, f"{osp.splitext(osp.basename(f))[0]}.npz")
        for f in dta_files
    ]

    metafunc.parametrize("dta_file, ref_file", list(zip(dta_files, ref_files)))
