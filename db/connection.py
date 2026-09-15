import os
from contextlib import contextmanager
from psycopg2 import pool as pg_pool
from dotenv import load_dotenv

load_dotenv()

db_host = os.environ["DATABASE_HOST"]
db_user = os.environ["DATABASE_USER"]
db_name = os.environ["DATABASE_NAME"]
db_password = os.environ["DATABASE_PASSWORD"]
db_port = os.environ["DATABASE_PORT"]

_min_conn = int(os.environ.get("DATABASE_POOL_MIN", "1"))
_max_conn = int(os.environ.get("DATABASE_POOL_MAX", "10"))

connection_pool = pg_pool.ThreadedConnectionPool(
    _min_conn,
    _max_conn,
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host,
    port=db_port,
    sslmode=os.environ.get("DATABASE_SSLMODE", "require"),
    # ponytail: default stays "require" so nothing weakens in prod; local docker
    # Postgres needs DATABASE_SSLMODE=disable in .env
)


@contextmanager
def get_conn():
    """Borrow a connection from the pool and always return it, even on error.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
    """
    conn = connection_pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        connection_pool.putconn(conn)


def close_pool() -> None:
    """Call on FastAPI shutdown to release all pooled connections cleanly."""
    connection_pool.closeall()
