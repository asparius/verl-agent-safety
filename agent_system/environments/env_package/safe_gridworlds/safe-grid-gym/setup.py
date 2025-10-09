try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    import setuptools

setuptools.setup(
    name="safe-grid-gym",
    version="0.1",
    description="A gym interface for AI safety gridworlds created in pycolab.",
    long_description=(
        "Provides an OpenAI Gym interface for the AI safety gridworlds created "
        "by DeepMind. This allows to train reinforcement learning agents that "
        "use the OpenAI Gym interface on the gridworld environments."
    ),
    url="https://github.com/david-lindner/safe-grid-gym/",
    author="David Lindner",
    author_email="dev@davidlindner.me",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
        "ai "
        "artificial intelligence "
        "gridworld "
        "gym "
        "rl "
        "reinforcement learning "
    ),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.28.0",  # Modern replacement for gym
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "matplotlib>=3.5.0",
    ],
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
    test_suite="safe_grid_gym.tests",
    package_data={"safe_grid_gym.envs.common": ["*.ttf"]},
)
