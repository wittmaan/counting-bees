from setuptools import setup, find_packages

setup(
    name="counting_bees",
    version="0.1",
    description="application for counting bees in video files",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "pandas",
        "pytest",
        "numpy",
        "torch",
        "black",
        "torchvision",
        "scikit-learn",
        "albumentations",
        "opencv-python",
    ],
)
