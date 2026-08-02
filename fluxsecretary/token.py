"""Credentials for the agent backend"""

from __future__ import annotations

import os

DEFAULT_TOKEN_FILE = "/etc/flux-secretary/token"

TOKEN_ENV = {
    "aws": "AWS_BEARER_TOKEN_BEDROCK",
    "claude": "ANTHROPIC_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

TOKEN_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "FLUX_SECRETARY_TOKEN",
)


def read_token(path=None):
    """Token from a mounted secret file, else the environment."""
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
    """Put the token where the SDK will look for it."""
    var = override or TOKEN_ENV.get(backend)
    if not var:
        for v in set(TOKEN_ENV.values()):
            os.environ.setdefault(v, token)
        return "all"
    os.environ[var] = token
    return var


def resolve_backend(backend, token_source=""):
    """Turn "auto" into a backend behalf actually knows."""
    if backend and backend != "auto":
        return backend
    for var, name in (
        ("AWS_BEARER_TOKEN_BEDROCK", "aws"),
        ("ANTHROPIC_API_KEY", "claude"),
        ("GOOGLE_API_KEY", "gemini"),
    ):
        if os.environ.get(var):
            return name
    return "aws" if token_source.startswith("file:") else ""
