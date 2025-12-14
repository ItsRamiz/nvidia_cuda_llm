import psycopg

def init_db():
    conn = psycopg.connect(
        "host=localhost port=5432 dbname=vectordb user=postgres password=postgres"
    )
    cur = conn.cursor()

    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR(3072)
    );
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    #init_db()
    print("Table already created!")


