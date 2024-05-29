"""This module contains the ARM toolchain."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def arm_workspace():
    http_archive(
        name = "arm-none-eabi",
        sha256 = "826353d45e7fbaa9b87c514e7c758a82f349cb7fc3fd949423687671539b29cf",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi",
        url = "https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi.tar.xz?rev=0f93cc5b9df1473dabc1f39b06feb468&hash=C32035997FC5C4F299BC61A85A09A4F86E2135D4",
    )
