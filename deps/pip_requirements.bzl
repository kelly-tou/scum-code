"""This module parses the pip dependencies."""

load("@rules_python//python:pip.bzl", "pip_parse")

def parse_pip_requirements():
    pip_parse(
        name = "pip_deps",
        python_interpreter_target = "@python3_12_host//:python",
        requirements_lock = "//deps:pip_requirements.txt",
    )
