"""MODULE 10 - the HTTP surface.

Liveness and readiness are separate endpoints answering separate questions,
errors carry stable codes rather than tracebacks, and nothing here writes an
image to disk.
"""

from hamqadam_ai.api.app import create_app, get_state

__all__ = ["create_app", "get_state"]
