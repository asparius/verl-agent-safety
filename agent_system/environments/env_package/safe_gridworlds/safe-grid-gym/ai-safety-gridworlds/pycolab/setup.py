# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

from pathlib import Path
from setuptools import setup, find_packages
import warnings

# Warn user about missing curses
try:
    import curses
except ImportError:
    warnings.warn(
        'The human_ui module and example games require the curses library. '
        'Without curses, pycolab can still be used as a library, '
        'but you cannot play games on the console.'
    )

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="pycolab",
    version="1.2.0.dev0",  # Development version
    description="An engine for small games for reinforcement learning agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/pycolab/",
    author="The pycolab authors",
    author_email="pycolab@deepmind.com",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Console :: Curses",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment :: Arcade",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Testing",
    ],
    keywords=(
        "ai ascii art game engine gridworld reinforcement learning retro retrogaming"
    ),
    python_requires=">=3.8, <3.13",
    install_requires=[
        "numpy>=1.9",
        "six",
    ],
    extras_require={
        "ndimage": ["scipy>=0.13.3"],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ],
    },
    packages=find_packages(),
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "aperture = pycolab.examples.aperture:main",
            "apprehend = pycolab.examples.apprehend:main",
            "extraterrestrial_marauders = pycolab.examples.extraterrestrial_marauders:main",
            "fluvial_natation = pycolab.examples.fluvial_natation:main",
            "hello_world = pycolab.examples.hello_world:main",
            "scrolly_maze = pycolab.examples.scrolly_maze:main",
            "shockwave = pycolab.examples.shockwave:main [ndimage]",
            "warehouse_manager = pycolab.examples.warehouse_manager:main",
            "chain_walk = pycolab.examples.classics.chain_walk:main",
            "cliff_walk = pycolab.examples.classics.cliff_walk:main",
            "four_rooms = pycolab.examples.classics.four_rooms:main",
        ],
    },
)

