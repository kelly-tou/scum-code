"""This module contains Bazel rules for creating packages."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PKG_VERSION = "1.0.0"

def rules_pkg_workspace():
    http_archive(
        name = "rules_pkg",
        sha256 = "cc1d6f58eb9bc2bfad247b20f07725dda2d6b119b62b11f1dab9a094a24222e6",
        strip_prefix = "rules_pkg-{}".format(RULES_PKG_VERSION),
        url = "https://github.com/bazelbuild/rules_pkg/archive/refs/tags/{}.tar.gz".format(RULES_PKG_VERSION),
    )
