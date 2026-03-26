import psycopg

DB_CONFIG = {
    "host": "0.tcp.ap.ngrok.io",
    "port": 10042,
    "dbname": "face_db",
    "user": "beam",
    "password": "123"
}

def get_connection():
    return psycopg.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    ) 

# port": must be updated every day to enter the database through ngrok
# please check the port number in the terminal after running ngrok