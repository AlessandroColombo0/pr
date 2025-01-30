from setuptools import setup, find_packages

setup(
    name="pr",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'asttokens>=3.0.0',
        'colorama>=0.4.6',
        'executing>=2.2.0',
        'numpy>=2.2.2',
        'pandas>=2.2.3',
        'Pygments>=2.19.1',
        'python-dateutil>=2.9.0',
        'pytz>=2024.2',
        'regex>=2024.11.6',
        'setuptools>=75.8.0',
        'six>=1.17.0',
        'tzdata>=2025.1'
    ]
)