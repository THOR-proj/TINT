#!/usr/bin/env python
# adapted from https://github.com/ARM-DOE/pyart/blob/master/setup.py
"""TINT: TINT is not TITAN

TINT is a package for doing TITAN style storm cell tracking using
pyart grid objects.

"""

import glob
import os
import sys
import subprocess

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins


DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Atmospheric Science
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

NAME = "tint"
AUTHORS = "Mark Picel, Bhupendra Raut"
MAINTAINER = "ARM Developers"
# MAINTAINER_EMAIL = ""
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
URL = "https://github.com/openradar/TINT.git"
DOWNLOAD_URL = "https://github.com/openradar/TINT.git"
LICENSE = "BSD"
CLASSIFIERS = filter(None, CLASSIFIERS.split("\n"))
PLATFORMS = ["Linux", "Mac OS-X", "Unix"]
MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)
SCRIPTS = glob.glob("scripts/*")


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")

# This is a bit hackish: we are setting a global variable so that the main
# tint __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet. While ugly, it's
# a lot more robust than what was previously being used.
builtins.__TINT_SETUP__ = True


def write_version_py(filename="tint/version.py"):
    cnt = """
# THIS FILE IS GENERATED FROM TINT SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of tint.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists(".git"):
        GIT_REVISION = git_version()
    elif os.path.exists("tint/version.py"):
        # must be a source distribution, use existing version file
        try:
            from tint.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError(
                "Unable to import git_revision. Try removing "
                "tint/version.py and the build directory "
                "before building."
            )
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += ".dev+" + GIT_REVISION[:7]

    a = open(filename, "w")
    try:
        a.write(
            cnt
            % {
                "version": VERSION,
                "full_version": FULLVERSION,
                "git_revision": GIT_REVISION,
                "isrelease": str(ISRELEASED),
            }
        )
    finally:
        a.close()


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage("tint")
    return config


def setup_package():

    # rewrite version file
    write_version_py()

    from numpy.distutils.core import setup

    setup(
        name=NAME,
        author=AUTHORS,
        maintainer=MAINTAINER,
        # maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        configuration=configuration,
        scripts=SCRIPTS,
    )


if __name__ == "__main__":
    setup_package()
