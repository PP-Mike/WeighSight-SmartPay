"""初始化商品库 fruit_vegetables.db（13 类，价格可自行修改）。

用法：
    python3 tools/init_db.py
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "fruit_vegetables.db"

# (英文名, 中文名, 单价 元/kg)
PRODUCTS = [
    ("Bitter melon", "苦瓜", 8.00),
    ("Carrot", "胡萝卜", 4.00),
    ("Green apple", "青苹果", 8.00),
    ("Lemon", "柠檬", 10.00),
    ("Pomegranate", "石榴", 12.00),
    ("Tomato", "番茄", 5.00),
    ("apple", "苹果", 8.00),
    ("eggplant", "茄子", 6.00),
    ("orange", "橙子", 7.00),
    ("peach", "桃子", 10.00),
    ("pear", "梨", 8.00),
    ("pumpkin", "南瓜", 4.00),
    ("watermelon", "西瓜", 3.00),
]


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name_en TEXT NOT NULL UNIQUE,
            name_cn TEXT NOT NULL,
            price REAL NOT NULL,
            unit TEXT DEFAULT 'kg',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            product_name TEXT,
            weight REAL,
            price REAL,
            amount REAL,
            image_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    added = 0
    for en, cn, price in PRODUCTS:
        cur.execute(
            "INSERT OR IGNORE INTO products (name_en, name_cn, price) VALUES (?, ?, ?)",
            (en, cn, price),
        )
        added += cur.rowcount

    conn.commit()
    total = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    conn.close()

    print(f"数据库: {DB}")
    print(f"本次新增 {added} 条，当前共 {total} 条商品")
    print("如需改价，可直接编辑数据库或使用程序内「管理商品库」功能。")


if __name__ == "__main__":
    main()
