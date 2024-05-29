"""This module contains Bazel rules for creating packages."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PKG_VERSION = "0.10.1"

def rules_pkg_workspace():
    http_archive(
        name = "rules_pkg",
        sha256 = "d330dbe3e3004241ddb9b377416ffc5c823e3e2c08c0d56a7e1935499e7f8577",
        strip_prefix = "rules_pkg-{}".format(RULES_PKG_VERSION),
        url = "https://github.com/bazelbuild/rules_pkg/archive/refs/tags/{}.tar.gz".format(RULES_PKG_VERSION),
    )
