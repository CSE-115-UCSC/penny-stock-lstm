
db = '../historical.db'
sqliteConnection = sqlite3.connect(db)
dbCursor = sqliteConnection.cursor()
print(f'SQlite connected with {db}')
