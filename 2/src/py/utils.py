from math import isnan

__author__ = 'Dmitry Spikhalskiy <dmitry@spikhalskiy.com>'

def default_if_nan(s, default=""):
    return default if isinstance(s, float) and isnan(s) else s
