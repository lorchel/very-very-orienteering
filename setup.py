#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: setup.py
#  Purpose: Installation script for Very-Very-Orienteering Map Generator
#   Author: Tom Richter
#    Email: lorchel@gmx.de
#  License: GPLv3
#
# Copyright (C) 2012 Tom Richter
#---------------------------------------------------------------------
"""
Very-Very-Orienteering Map Generator
====================================

This is a Very-Very-Orienteering Map Generator. If you don't know these kind of
orienteering training: Try it out! It's fun!
"""

import distribute_setup
# automatically install distribute if the user does not have it installed
distribute_setup.use_setuptools()
from setuptools import setup
import os
import shutil
import sys


LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
DOCSTRING = __doc__.split('\n')
# package specific settings
VERSION = '0.1'
NAME = 'very_very_orienteering'
AUTHOR = 'Tom Richter'
AUTHOR_EMAIL = 'lorchel@gmx.de'
LICENSE = 'GNU General Public License, Version 3 (GPLv3)'
KEYWORDS = ['very-very-orienteering', 'orienteering', 'map generator']
INSTALL_REQUIRES = ['setuptools', 'numpy', 'matplotlib', 'scipy', 'PIL']
if sys.version_info.major == 3:
    sys.exit('This package is not supporting python3 because of dependencies '
             'which are not migrated.')
    ENTRY_POINTS = {'console_scripts': ['very-very-o3 = very_very_orienteering:main']}
else:
    ENTRY_POINTS = {'console_scripts': ['very-very-o = very_very_orienteering:main']}


def convert2to3():
    """
    Convert source to Python 3.x syntax using lib2to3.
    """
    # create a new 2to3 directory for converted source files
    dst_path = os.path.join(LOCAL_PATH, '2to3')
    shutil.rmtree(dst_path, ignore_errors=True)
    # copy original tree into 2to3 folder ignoring some unneeded files

    def ignored_files(adir, filenames):  # @UnusedVariable
        return ['.git', '2to3', 'build', 'dist'] + \
               [fn for fn in filenames if fn.startswith('distribute')] + \
               [fn for fn in filenames if fn.endswith('.egg-info')]
    shutil.copytree(LOCAL_PATH, dst_path, ignore=ignored_files)
    os.chdir(dst_path)
    sys.path.insert(0, dst_path)
    # run lib2to3 script on duplicated source
    from lib2to3.main import main
    print('Converting to Python3 via lib2to3...')
    main('lib2to3.fixes', ['-w', '-n', '--no-diffs', 'smsl.py'])

def setupPackage():
    # use lib2to3 for Python 3.x
    if sys.version_info.major == 3:
        convert2to3()
    # setup package
    setup(
        name=NAME,
        version=VERSION,
        description=DOCSTRING[1],
        long_description=' '.join(DOCSTRING[4:-1]),
        url='',
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        platforms='OS Independent',
        classifiers=[
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Sport :: Orienteering'],
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        py_modules=['very_very_orienteering'],
    )
    # cleanup after using lib2to3 for Python 3.x
    if sys.version_info.major == 3:
        os.chdir(LOCAL_PATH)


if __name__ == '__main__':
    setupPackage()
