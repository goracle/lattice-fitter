"""Pylint-friendly import of profile, from line-profiler tool."""
try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    PROFILE = lambda x: x   # if it's not defined simply ignore the decorator.
