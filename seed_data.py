import sqlite3

# Connect to database
conn = sqlite3.connect("test_stock.db")
cursor = conn.cursor()

# Reset and recreate table with rich data
cursor.execute("DROP TABLE IF EXISTS products")
cursor.execute("""
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    stock INTEGER NOT NULL,
    price REAL NOT NULL
)
""")

# Sample inventory items
sample_items = [
    ("Filtre à huile", 12, 150.0),
    ("Plaquettes de frein", 3, 320.0),      # Low stock item
    ("Huile moteur 5W40", 25, 450.0),
    ("Batterie 12V 70Ah", 2, 950.0),        # High price & low stock
    ("Bougies d'allumage", 40, 60.0),
    ("Liquide de refroidissement", 0, 85.0) # Out of stock item
]

cursor.executemany("INSERT INTO products (name, stock, price) VALUES (?, ?, ?)", sample_items)
conn.commit()
conn.close()

print("Mock inventory updated successfully!")