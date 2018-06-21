"""Pylint-friendly import of profile, from line-profiler tool."""


try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(fnx):
        """line profiler default."""
        return fnx   # if it's not defined simply ignore the decorator.
# PROFILE = lambda x: x
