from setuptools import setup, find_packages

setup(
    name='ai-safety-gridworlds',
    version='0.1.0',
    description='AI Safety Gridworlds - A suite of reinforcement learning environments',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.0',
        'absl-py>=0.7.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pylint',
        ],
    },
    author='DeepMind',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
