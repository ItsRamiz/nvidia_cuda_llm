import psycopg

def test_connection():
    conn = psycopg.connect(
        "host=localhost port=5432 dbname=vectordb user=postgres password=postgres"
    )

    cur = conn.cursor()
    cur.execute("SELECT 1")
    print(cur.fetchone())

    conn.close()
