from setuptools import setup, find_packages

setup(
    name='JustinScoreCardPy',
    packages=find_packages(),
    version='0.0.2',
    install_requires=[
        'numpy >= 1.20',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'xgboost',
        'statsmodels',
        'SciencePlots'
    ]
)

