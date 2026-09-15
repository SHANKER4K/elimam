from db.connection import connection_pool


conn = connection_pool.getconn()

with conn.cursor() as cur:
    cur.execute("SELECT version();")
    print(cur.fetchone())
