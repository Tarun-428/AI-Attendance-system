import mysql.connector
import os

class Database:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=os.environ.get("DB_HOST"),
            port=int(os.environ.get("DB_PORT")),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASS"),
            database=os.environ.get("DB_NAME")
        )
        self.cursor = self.conn.cursor(dictionary=True)

    def create(self, query, params=None):
        self.cursor.execute(query, params or ())
        self.conn.commit()

    def insert(self, query, params):
        self.cursor.execute(query, params)
        self.conn.commit()
        return self.cursor.lastrowid

    def read(self, query, params=None):
        self.cursor.execute(query, params or ())
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()

# Initialize the database connection
conn = Database()
