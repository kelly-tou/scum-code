"""This module contains Bazel rules for Python."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PYTHON_VERSION = "1.0.0"

def rules_python_workspace():
    http_archive(
        name = "rules_python",
        sha256 = "4f7e2aa1eb9aa722d96498f5ef514f426c1f55161c3c9ae628c857a7128ceb07",
        strip_prefix = "rules_python-{}".format(RULES_PYTHON_VERSION),
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/{}.tar.gz".format(RULES_PYTHON_VERSION),
    )
