"""This module loads all third-party workspaces."""

load("//third_party/arm:workspace.bzl", "arm_workspace")
load("//third_party/rules_pkg:workspace.bzl", "rules_pkg_workspace")
load("//third_party/rules_python:workspace.bzl", "rules_python_workspace")

def load_third_party_workspaces():
    arm_workspace()
    rules_pkg_workspace()
    rules_python_workspace()
