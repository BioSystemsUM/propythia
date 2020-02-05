from setuptools import setup, find_packages
package_data={'propythia':['adjuv_functions/features_functions/data/*','adjuv_functions/sequence/data/*']}
# package_data={
#     'propythia.adjuv_functions.features_functions.data': ['*'],     # All files from folder A
#     'propythia.adjuv_functions.sequence.data': ['*']  #All text files from folder B
# },
setup(
    name = 'propythia',
    version = '0.0.4',
    package_dir = {'':'src'},
    packages = find_packages('src'),
    package_data = package_data,
    include_package_data=False,
    install_requires = ["numpy",
                        "scipy",
                        "pandas",
                        "matplotlib",
                        "sklearn",
                        "biopython"],
    author = 'Ana Marta Sequeira',
    author_email = 'anamartasequeira94@gmail.com',
    description = 'propythia - automated platform for the classification of peptides/proteins using machine learning',
    license = 'GNU General Public License v3',
    keywords = 'machine learning classification proteins',
    url = 'https://github.com/BioSystemsUM/propythia/archive/v_0.0.4.tar.gz',
    long_description = open('README.rst').read(),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
