"""
Compute-platform provenance for Icechunk commit metadata.

Every commit made by the processing pipeline embeds a small provenance
dictionary describing where and how the work ran (GitHub Actions runner,
CryoCloud/JupyterHub session, or a local machine), so that any
tile x water_year in the store can be traced back to the environment that
produced it. See the root README, "Rebuild processing pipeline on icechunk".
"""

import getpass
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict


def _package_versions() -> Dict[str, str]:
    """Versions of the packages that most affect processing results."""
    versions = {}
    for pkg in ("icechunk", "zarr", "xarray", "odc-stac", "odc-geo", "dask", "easysnowdata", "numpy"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _git_sha() -> str:
    """Git SHA of the processing code, from GitHub Actions env or local git."""
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def collect_provenance() -> Dict[str, Any]:
    """
    Collect a JSON-serializable description of the compute platform.

    Detection order:
    1. GitHub Actions (GITHUB_ACTIONS=true): runner + workflow/run identifiers,
       so a commit can be traced to the exact Actions run and attempt.
    2. JupyterHub-based platforms such as CryoCloud (JUPYTERHUB_USER set):
       hub user and image.
    3. Anything else: plain hostname/OS/user.

    Returns:
        Dictionary safe to pass as (part of) icechunk commit metadata.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        prov: Dict[str, Any] = {
            "platform": "github-actions",
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "job": os.getenv("GITHUB_JOB"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "runner_os": os.getenv("RUNNER_OS"),
            "runner_arch": os.getenv("RUNNER_ARCH"),
        }
    elif os.getenv("JUPYTERHUB_USER"):
        prov = {
            "platform": "jupyterhub",  # e.g. CryoCloud
            "hub_host": os.getenv("JUPYTERHUB_HOST") or os.getenv("JUPYTERHUB_API_URL"),
            "hub_user": os.getenv("JUPYTERHUB_USER"),
            "image": os.getenv("JUPYTER_IMAGE") or os.getenv("JUPYTER_IMAGE_SPEC"),
            "hostname": platform.node(),
        }
    else:
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        prov = {
            "platform": "local",
            "hostname": platform.node(),
            "os": platform.platform(),
            "user": user,
        }

    prov["python"] = platform.python_version()
    prov["code_sha"] = _git_sha()
    prov["package_versions"] = _package_versions()
    return {k: v for k, v in prov.items() if v is not None}
