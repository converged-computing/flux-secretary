"""Credentials for the agent backend
No Flux here. Reading and placing a credential has nothing to do with the
allocation, and keeping it separate means it can be tested without a broker.
"""

from __future__ import annotations

import os

DEFAULT_TOKEN_FILE = "/etc/flux-secretary/token"

# Which environment variable a backend authenticates from. A token read out of a
# mounted file is not enough on its own: the SDKs read the environment, so the
# value has to be put back there before a runner is built.
TOKEN_ENV = {
    "aws": "AWS_BEARER_TOKEN_BEDROCK",
    "claude": "ANTHROPIC_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# Checked in order when no file is mounted.
TOKEN_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "FLUX_SECRETARY_TOKEN",
)


def read_token(path=None):
    """Token from a mounted secret file, else the environment.

    The MiniCluster CRD takes environment as plain key/value with no
    secretKeyRef, so a mounted secret is how a credential gets into the pod.
    """
    for p in (path, DEFAULT_TOKEN_FILE):
        if p and os.path.isfile(p):
            tok = open(p).read().strip()
            if tok:
                return tok, f"file:{p}"
    for var in TOKEN_VARS:
        if os.environ.get(var):
            return os.environ[var], f"env:{var}"
    return None, "none"


def export_token(token, backend, override=""):
    """Put the token where the SDK will look for it.

    A file mounted at /etc/flux-secretary/token means nothing to boto3 or the
    Anthropic client. They read the environment, so the value is exported under
    the variable the chosen backend authenticates from.
    """
    var = override or TOKEN_ENV.get(backend)
    if not var:
        # Backend not yet resolved (auto): set every one we recognise rather
        # than guess wrong and fall back to deterministic for no reason.
        for v in set(TOKEN_ENV.values()):
            os.environ.setdefault(v, token)
        return "all"
    os.environ[var] = token
    return var


def resolve_backend(backend, token_source=""):
    """Turn "auto" into a backend behalf actually knows.

    behalf accepts claude, gemini or aws. "auto" is not one of them, and passing
    it through made behalf exit, which ended the run instead of falling back.
    """
    if backend and backend != "auto":
        return backend
    for var, name in (
        ("AWS_BEARER_TOKEN_BEDROCK", "aws"),
        ("ANTHROPIC_API_KEY", "claude"),
        ("GOOGLE_API_KEY", "gemini"),
    ):
        if os.environ.get(var):
            return name
    # A mounted token says nothing about which service it belongs to. Bedrock is
    # the common case here, and --backend makes it explicit.
    return "aws" if token_source.startswith("file:") else ""
