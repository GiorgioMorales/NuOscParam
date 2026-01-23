import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='NuOscParam',
    version='0.0.1',
    author='Giorgio Morales - GREYC',
    author_email='giorgiomorales@ieee.org',
    description='Neutrino Parameter Estimation from Oscillation Probability Maps',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/GiorgioMorales/NuOscParam',
    project_urls={"Bug Tracker": "https://github.com/GiorgioMorales/NuOscParam/issues"},
    license='MIT',
    packages=setuptools.find_packages('src', exclude=['test']),
    # packages=setuptools.find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=['numpy', 'opencv-python', 'tqdm', 'h5py', 'pyodbc', 'regex', 'emcee',
                      'torchsummary', 'python-dotenv', 'omegaconf', 'pandas', 'pynvml', 'matplotlib'],
)
