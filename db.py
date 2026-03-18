import psycopg

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "face_attendance",
    "user": "face_user",
    "password": "1234"
}

def get_connection():
    return psycopg.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )