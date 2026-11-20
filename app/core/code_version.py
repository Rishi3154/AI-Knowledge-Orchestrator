"""
code_version.py

A fingerprint of the app's own source code.

Streamlit Community Cloud applies a git push to a running app without
restarting it: changed modules are re-imported, but objects kept by
st.cache_resource survive. A pipeline built by the old code then
keeps running next to the new code and can disagree with it (after
the switch to Claude it was still reading the old storage collection,
so new uploads never appeared). Passing this fingerprint as an
argument to cached builders makes any code change build fresh ones.
"""

import hashlib
from itertools import chain

from app.core.config import PROJECT_ROOT

SOURCE_DIRS = ("app", "components")


def code_version() -> str:
    """
    Changes whenever a Python file of the app is added, removed or
    modified. Uses file sizes and modification times, so it costs a
    few dozen file stats and no reading.
    """

    files = chain(
        [PROJECT_ROOT / "app.py"],
        *(
            (PROJECT_ROOT / folder).rglob("*.py")
            for folder in SOURCE_DIRS
        ),
    )

    digest = hashlib.sha1()

    for path in sorted(files):

        try:
            stat = path.stat()
        except FileNotFoundError:
            continue

        digest.update(
            f"{path.relative_to(PROJECT_ROOT)}:"
            f"{stat.st_mtime_ns}:{stat.st_size}\n".encode()
        )

    return digest.hexdigest()
