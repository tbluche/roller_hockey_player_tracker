import os
from setuptools import setup, find_packages

PACKAGE_NAME = "rhtrack"
VERSION = "0.1.0"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)


def package_files(directory, subdir):
    paths = []
    for (path, _, filenames) in os.walk(os.path.join(directory, subdir)):
        for filename in filenames:
            paths.append(os.path.join(path.replace(directory, ""), filename))
    return paths


install_requires = [
    "torch",
    "torchvision",
    "opencv-contrib-python",
    "tqdm",
    "youtube-dl",
    "numpy",
    "matplotlib",
    "sklearn",
]


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="Theodore Bluche",
    license="",
    install_requires=install_requires,
    packages=find_packages(),
    package_dir={PACKAGE_NAME: PACKAGE_NAME},
    package_data={"": []},
    include_package_data=True,
    zip_safe=False,
    command_options={},
)
