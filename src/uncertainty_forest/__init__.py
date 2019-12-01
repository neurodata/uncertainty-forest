# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
name = "uncertainty_forest"

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'uncertainty-forest'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
