import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": int(os.environ["DB_PORT"]),
    "database": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

CSV_DIR = os.getenv("CSV_DIR", "data")


class DatabaseUtil:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = psycopg2.connect(**db_config)

    def schema_details(self, schema_name):
        schema_info = [f"Schema: {schema_name}\n"]

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name;
                """,
                (schema_name,),
            )
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                schema_info.append(f"\nTable: {table_name}\n")
                cursor.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (schema_name, table_name),
                )

                for column_name, data_type in cursor.fetchall():
                    schema_info.append(
                        f"  Column: {column_name}, Data Type: {data_type}\n"
                    )

                cursor.execute(
                    sql.SQL("SELECT * FROM {}.{} LIMIT 5;").format(
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name),
                    )
                )
                schema_info.append(f"Sample Data from {table_name}:\n")
                schema_info.extend(f"  {row}\n" for row in cursor.fetchall())

        return "".join(schema_info)

    def close(self):
        if self.connection and not self.connection.closed:
            self.connection.close()


#def main():
#    database = DatabaseUtil(DB_CONFIG)
#    try:
#        schema_info = database.schema_details("public")
#        output_path = PROJECT_ROOT / "schema_info.txt"
#        output_path.write_text(schema_info, encoding="utf-8")
#        print(f"Schema information written to {output_path}")
#    finally:
#        database.close()
#
#
#if __name__ == "__main__":
#    main()
#
