"""Eval box transport — vast.ai (default) or a fixed SSH box.

Set EVAL_TRANSPORT=ssh to use a pinned bare-metal GPU (EVAL_SSH_HOST / EVAL_SSH_PORT).
Leave unset or EVAL_TRANSPORT=vast to reuse a pinned vast.ai instance (no auto-rent/stop).

Legacy alias: EVAL_USE_VAST=0 also selects the SSH box (requires EVAL_SSH_HOST).
"""

import os

_TRANSPORT = os.environ.get("EVAL_TRANSPORT", "").strip().lower()


def vast_enabled():
    """True when vast.ai provisioning should run (the default)."""
    if _TRANSPORT == "ssh":
        return False
    if _TRANSPORT == "vast":
        return True
    use_vast = os.environ.get("EVAL_USE_VAST", "").strip().lower()
    if use_vast in ("0", "false", "no"):
        return False
    return True


def ssh_box_endpoint():
    """Return (host, port) for the fixed SSH eval box, or None if not configured."""
    direct = os.environ.get("EVAL_SSH", "").strip()
    if direct:
        host, sep, port = direct.partition(":")
        if not host:
            return None
        return host.strip(), int(port or "22")
    host = os.environ.get("EVAL_SSH_HOST", "").strip()
    if not host:
        return None
    return host, int(os.environ.get("EVAL_SSH_PORT", "22"))


def ssh_box_arg():
    """host:port string for vast_eval.py --ssh, or empty."""
    ep = ssh_box_endpoint()
    if not ep:
        return ""
    host, port = ep
    return f"{host}:{port}"


def ssh_box_enabled():
    """Use the fixed SSH box instead of vast.ai."""
    if vast_enabled():
        return False
    return ssh_box_endpoint() is not None


def ssh_box_user():
    """Remote SSH user for the fixed box. vast.ai images run as root, so that's the
    default; a bare-metal box with a non-root account (e.g. cloud-init default users)
    overrides via EVAL_SSH_USER. Only meaningful when ssh_box_enabled()."""
    return os.environ.get("EVAL_SSH_USER", "root").strip() or "root"
