"""This module parses the pip dependencies."""

load("@python3_12//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

def parse_pip_requirements():
    pip_parse(
        name = "pip_deps",
        python_interpreter_target = interpreter,
        requirements_lock = "//deps:pip_requirements.txt",
    )
