"""Pylint-friendly import of profile, from line-profiler tool."""


try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def PROFILE(x):
        """line profiler default."""
        return x   # if it's not defined simply ignore the decorator.
# PROFILE = lambda x: x
