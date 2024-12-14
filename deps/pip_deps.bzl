"""This module loads the pip dependencies."""

load("@pip_deps//:requirements.bzl", "install_deps")

def load_pip_dependencies():
    install_deps(
        python_interpreter_target = "@python3_12_host//:python",
    )
