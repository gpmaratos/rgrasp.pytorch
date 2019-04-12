from distutils.core import setup, Extension

module1 = Extension('pcdreader',
    sources = ['read_pcd.cpp'],
    include_dirs = ['/usr/lib/python3.7/site-packages/numpy/core/include/']
)

setup(name = 'pcdreader',
    version = '1.0',
    description = 'Read a point cloud file into numpy array.',
    ext_modules = [module1]
)
