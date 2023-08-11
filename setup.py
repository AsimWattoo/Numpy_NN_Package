from setuptools import setup

setup(
    name='numpy_nn',
    version='0.1.1',    
    description='A custom neural network implementation',
    url='https://github.com/AsimWattoo/Numpy_NN_Package.git',
    author='Muhammad Asim',
    author_email='asimwattoo6@gmail.com',
    license='BSD 2-clause',
    packages=['numpy_nn'],
    install_requires=['pandas',
                      'numpy',
                      'matplotlib'                     
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: Freeware', 
        'Natural Language :: English',
    ],
)