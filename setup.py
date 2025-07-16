from setuptools import setup, find_packages

setup(
    name='scoresight',
    version='1.0.0',
    author='Your Name',
    description='A machine learning pipeline for predicting student academic scores using linear regression.',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.3',
        'numpy>=1.26.4',
        'matplotlib>=3.8.4',
        'seaborn>=0.13.2',
        'scikit-learn>=1.4.2',
        'joblib>=1.4.2'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
