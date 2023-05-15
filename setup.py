from setuptools import setup

setup(
    name='nwp-data-loader',
    version='0.1.0',    
    description='A example Python package',
    url='https://github.com/shuds13/pyexample',
    author='Bobby Antonio',
    author_email='bobbyantonio@gmail.com',
    license='GNU GENERAL PUBLIC LICENSE v3.0',
    packages=['nwpdl'],
    install_requires=['netCDF4', 
                      'tqdm',
                      'numpy', 
                      'pandas', 
                      'xarray', 
                      'xesmf'                    
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: GNU General Public License',  
        'Operating System :: POSIX :: Linux',     
        'Programming Language :: Python :: 3.9',
    ],
)