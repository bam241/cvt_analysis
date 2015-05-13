import sqlite3

db = sqlite3.connect(':memory:')
db = sqlite3.connect('data/random_sink/HEU03p_aq33_s05_swu180_t100_E5.sqlite')

cursor = db.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

test = cursor.execute("PRAGMA table_info('Inventories');")
print(cursor.fetchall())

for r in test:
  print r


db.close()