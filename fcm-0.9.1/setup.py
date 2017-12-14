from setuptools import setup, Extension
from numpy import get_include

logicle_extension = Extension('fcm.core._logicle',
                              sources = [ 'src/core/logicle_ext/%s' %i for i in [
                                'Logicle.cpp',
                                'my_logicle.cpp',
                                'my_logicle_wrapper.cpp']],
                              include_dirs = [get_include()]
                              )
setup(name='fcm',
      version='0.9.1',
      url='http://code.google.com/p/py-fcm/',
      packages=['fcm', 'fcm.core', 'fcm.graphics', 'fcm.gui', 'fcm.io', 'fcm.statistics' ],
      package_dir = {'fcm': 'src'},
      package_data= {'': ['data/*']},
      description='Python Flow Cytometry (FCM) Tools',
      author='Jacob Frelinger',
      author_email='jacob.frelinger@duke.edu',
      ext_modules = [logicle_extension],
      requires=['numpy (>=1.3.0)',
                'scipy (>=0.6)',
		'dpmix (>=0.1)',
                'matplotlib (>=1.0)'], # figure out the rest of what's a required package.
      )
