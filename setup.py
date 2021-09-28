from setuptools import setup, find_namespace_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='miniasr',

    version='0.1',

    description='A mini, simple, and fast end-to-end automatic speech recognition toolkit.',

    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vectominist/MiniASR',

    author='Heng-Jui Chang',

    author_email='b06901020@ntu.edu.tw',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'License :: OSI Approved :: MIT License',

        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='speech, speech recognition, ctc, asr',

    packages=find_namespace_packages(include=["miniasr*"]),

    python_requires='>=3.6, <4',

    install_requires=[
        "tqdm",
        "numpy>=1.19.5",
        "sentencepiece>=0.1.96",
        "pytorch-lightning>=1.3.8",
        "easydict",
        "joblib>=0.12.4",
        "librosa>=0.7.2",
        "numba==0.48",
        "edit_distance",
        "torch>=1.7.0",
        "torchaudio>=0.7.0",
        "torchvision>=0.8.0",
        "torchtext>=0.8.0",
    ],

    entry_points={
        "console_scripts": [
            "minasr-asr = run_asr:main",
            "miniasr-preprocess = run_preprocess:main",
        ],
    },

    project_urls={
        'Bug Reports': 'https://github.com/vectominist/MiniASR/issues',
        'Source': 'https://github.com/vectominist/MiniASR/',
    },
)
