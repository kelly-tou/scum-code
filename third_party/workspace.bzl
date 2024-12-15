"""This module loads all third-party workspaces."""

load("//third_party/arm:workspace.bzl", "arm_workspace")

def load_third_party_workspaces():
    arm_workspace()
