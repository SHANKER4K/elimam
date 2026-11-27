"""One way to turn a database error into an HTTP error.

Postgres messages name tables, constraints and the offending values, so they
are logged here (with the traceback) and the client gets a generic detail.
"""

import logging

from fastapi import HTTPException

logger = logging.getLogger("api.db")


def db_error(exc: Exception, status_code: int = 500) -> HTTPException:
    logger.error(
        "database_error",
        exc_info=exc,
        extra={"pgcode": getattr(exc, "pgcode", None)},
    )
    return HTTPException(status_code=status_code, detail="Database error")
