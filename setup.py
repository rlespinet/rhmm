import numpy
from numpy.distutils.core import setup, Extension

module = Extension('rhmm',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = [numpy.get_include(), 'python/',
                                    'lib/', 'dependencies/eigen/'],
                    language = "c++",
                    libraries = [],
                    library_dirs = [],
                    extra_compile_args=['-std=c++14',
                                        '-fopenmp',
                                        '-O0',
                                        '-g',
                                        '-Wno-write-strings'],
                    extra_link_args=[ '-fopenmp' ],
                    sources = [
                        'python/rhmm.cpp',
                        'python/hmm_class.cpp',
                        'python/distribution_class.cpp',
                        # 'lib/pca.cpp',
                        'lib/hmm.cpp',
                    ])

setup (name = 'RLEARNING',
       version = '1.0',
       description = 'Functions related to machine learning',
       author = 'Remi Lespinet',
       author_email = 'remi@lespi.net',
       url = '',
       long_description = '''
Functions related to machine learning
''',
       ext_modules = [module])
