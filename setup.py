from setuptools import setup, find_packages

package_data = {'propythia': ['adjuv_functions/features_functions/data/*', 'adjuv_functions/sequence/data/*']}
# package_data={
#     'propythia.adjuv_functions.features_functions.data': ['*'],     # All files from folder A
#     'propythia.adjuv_functions.sequence.data': ['*']  #All text files from folder B
# },
setup(
    name='propythia',
    version='2.0.0',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data=package_data,
    include_package_data=True,
    install_requires=[
        "biopython==1.80",
        "holoviews==1.15.3",
        "joblib==1.2.0",
        "keras==2.11.0",
        "matplotlib==3.6.2",
        "networkx==2.8.8",
        "numpy==1.23.5",
        "scikit-learn==1.2.0",
        "scipy==1.9.3",
        "seaborn==0.12.1",
        "tensorboard==2.11.0",
        "tensorflow==2.11.1",
        "tqdm==4.64.1",
        "umap==0.1.1",
        "umap-learn[plot]==0.5.3",
        "bokeh==2.4.3",
        "chardet==5.1.0"
    ],
    author='Ana Marta Sequeira',
    author_email='anamartasequeira94@gmail.com',
    description='propythia - automated platform for the classification of peptides/proteins using machine learning',
    license='GNU General Public License v3',
    keywords='machine learning deep learning classification proteins',
    url='https://github.com/BioSystemsUM/propythia',
    long_description=open('README.rst').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
