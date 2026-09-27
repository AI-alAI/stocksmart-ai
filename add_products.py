import sqlite3

DB_FILE = "test_stock.db"

# Products to add — (name, stock, price)
NEW_ITEMS = [
    ("Courroie de distribution", 7, 280.0),
    ("Filtre à air", 15, 120.0),
    ("Amortisseur avant", 1, 1250.0),
    ("Liquide de frein DOT4", 0, 95.0),
    ("Alternateur", 4, 2400.0),
    ("Démarreur", 6, 1800.0),
    ("Pompe à eau", 8, 640.0),
    ("Radiateur", 3, 1150.0),
    ("Pneu 195/65 R15", 20, 720.0),
    ("Essuie-glaces avant", 30, 90.0),
]


def main():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Ensure the table exists (in case the DB is brand new)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        stock INTEGER NOT NULL,
        price REAL NOT NULL
    )
    """)

    added, skipped = 0, 0
    for name, stock, price in NEW_ITEMS:
        cur.execute("SELECT 1 FROM products WHERE name = ?", (name,))
        if cur.fetchone():
            print(f"⏭️  Skipped (already exists): {name}")
            skipped += 1
        else:
            cur.execute(
                "INSERT INTO products (name, stock, price) VALUES (?, ?, ?)",
                (name, stock, price),
            )
            print(f"➕ Added: {name}  ({stock} pcs, {price:.2f} MAD)")
            added += 1

    conn.commit()

    # Show final count
    cur.execute("SELECT COUNT(*) FROM products")
    total = cur.fetchone()[0]
    conn.close()

    print()
    print(f"✅ Done. Added {added}, skipped {skipped}.")
    print(f"📦 Total products in DB: {total}")


if __name__ == "__main__":
    main()