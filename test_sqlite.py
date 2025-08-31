import sqlite3
conn = sqlite3.connect("test.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE test (id INTEGER, name TEXT)")
cursor.execute("INSERT INTO test VALUES (1, 'corrosion')")
conn.commit()
conn.close()
print("✅ SQLite is working!")