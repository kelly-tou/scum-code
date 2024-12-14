"""This module contains Bazel rules for creating packages."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PKG_VERSION = "1.0.1"

def rules_pkg_workspace():
    http_archive(
        name = "rules_pkg",
        sha256 = "23005750a27aabfd5975a3d5aeac9542371cbfa24d3ad74e47f80b84547754da",
        strip_prefix = "rules_pkg-{}".format(RULES_PKG_VERSION),
        url = "https://github.com/bazelbuild/rules_pkg/archive/refs/tags/{}.tar.gz".format(RULES_PKG_VERSION),
    )
