import os

import psycopg

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def connect_db():

    connection = psycopg.connect(DATABASE_URL)

    print("Database connected successfully.")

    return connection


def create_table():

    connection = connect_db()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS robot_data (
            id BIGSERIAL PRIMARY KEY,
            trait VARCHAR(50),
            axis_1 DOUBLE PRECISION,
            axis_2 DOUBLE PRECISION,
            axis_3 DOUBLE PRECISION,
            axis_4 DOUBLE PRECISION,
            axis_5 DOUBLE PRECISION,
            axis_6 DOUBLE PRECISION,
            axis_7 DOUBLE PRECISION,
            axis_8 DOUBLE PRECISION,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            time TIME,
            UNIQUE (year, month, day, time)
        )
    """)

    connection.commit()

    cursor.close()
    connection.close()

    print("robot_data table is ready.")


def insert_record(
    trait,
    axis_1,
    axis_2,
    axis_3,
    axis_4,
    axis_5,
    axis_6,
    axis_7,
    axis_8,
    timestamp
):

    from datetime import datetime

    timestamp = datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    )

    year = timestamp.year
    month = timestamp.month
    day = timestamp.day
    time = timestamp.time().replace(tzinfo=None)

    connection = connect_db()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO robot_data (
            trait,
            axis_1,
            axis_2,
            axis_3,
            axis_4,
            axis_5,
            axis_6,
            axis_7,
            axis_8,
            year,
            month,
            day,
            time
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (year, month, day, time) DO NOTHING
    """, (
        trait,
        axis_1,
        axis_2,
        axis_3,
        axis_4,
        axis_5,
        axis_6,
        axis_7,
        axis_8,
        year,
        month,
        day,
        time
    ))

    connection.commit()

    if cursor.rowcount == 1:
        print("Record inserted successfully.")
    else:
        print("Duplicate record detected. Record was not inserted.")

    cursor.close()
    connection.close()


def get_records():

    connection = connect_db()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM robot_data
        ORDER BY id
    """)

    records = cursor.fetchall()

    cursor.close()
    connection.close()

    print(f"{len(records)} record(s) retrieved from Neon.")

    return records