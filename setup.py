import os
import setuptools


def get_readme():
    with open("README.md", "r") as f:
        return f.read()


def generate_package_dir():
    dependencies = setuptools.find_packages("dependencies")
    # filter out submodules
    dependencies = [dependency for dependency in dependencies if "." not in dependency]
    package_dir = {dependency: os.path.join("dependencies", dependency) for dependency in dependencies}
    return package_dir


setuptools.setup(
    name="gcsl",
    version="0.1.0",
    description="Goal-Conditioned Supervised Learning",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.5.0",
    packages=setuptools.find_packages() + setuptools.find_packages("dependencies"),
    package_dir=generate_package_dir(),
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    install_requires=[
        "box2d-py",
        "cloudpickle",
        "gym",
        "ipython",
        "matplotlib",
        "mujoco_py",
        "numpy",
        "opencv-python",
        "Pillow",
        "torch",
        "tqdm",
        "transforms3d",
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ]
)
