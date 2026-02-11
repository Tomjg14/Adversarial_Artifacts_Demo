from setuptools import setup, find_packages

setup(
    name='adversarial-demo',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'numpy',
        'torch',
        'torchvision',
        'requests',
    ],
    entry_points={
        'console_scripts': [
            'run_demo=adversarial_artifacts.demo:start_adversarial_demo',
        ],
    },
)