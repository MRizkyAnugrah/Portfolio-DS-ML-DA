CREATE TABLE tokopaedi (
    row_id SERIAL PRIMARY KEY,
    order_id VARCHAR(50),
    order_date DATE,
    ship_date DATE,
    ship_mode VARCHAR(10),
    customer_id VARCHAR(50),
    customer_name VARCHAR(100),
    segment VARCHAR(20),
    country VARCHAR(100),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(20),
    region VARCHAR(100),
    product_id VARCHAR(50),
    category VARCHAR(50),
    subcategory VARCHAR(50),
    product_name VARCHAR(100),
    sales DECIMAL(10,2),
    quantity INTEGER,
    discount DECIMAL(5,2),
    profit DECIMAL(10,2)
);

INSERT INTO tokopaedi (order_id, order_date, ship_date, ship_mode, customer_id, customer_name, segment, country, city, state, postal_code, region, product_id, category, subcategory, product_name, sales, quantity, discount, profit)
VALUES
('ORD001', '2023-06-01', '2023-06-05', 'Reguler', 'CUST001', 'John Doe', 'Rumah', 'Indonesia', 'Jakarta', 'DKI Jakarta', '12345', 'Wilayah A', 'PROD001', 'Elektronik', 'Televisi', 'TV LED 32"', 500.00, 2, 10.00, 100.00),
('ORD002', '2023-06-02', '2023-06-06', 'Ekspress', 'CUST002', 'Jane Smith', 'Kantor', 'Indonesia', 'Surabaya', 'Jawa Timur', '67890', 'Wilayah B', 'PROD002', 'Pakaian', 'Baju', 'Kemeja Putih', 100.00, 3, 5.00, 50.00),
('ORD003', '2023-06-03', '2023-06-07', 'Reguler', 'CUST003', 'David Johnson', 'Rumah', 'Indonesia', 'Bandung', 'Jawa Barat', '54321', 'Wilayah A', 'PROD001', 'Elektronik', 'Televisi', 'TV LED 42"', 800.00, 1, 20.00, 200.00),
('ORD004', '2023-06-04', '2023-06-08', 'Reguler', 'CUST004', 'Sarah Brown', 'Kantor', 'Indonesia', 'Medan', 'Sumatera Utara', '98765', 'Wilayah C', 'PROD003', 'Makanan', 'Snack', 'Keripik Kentang', 50.00, 5, 2.50, 10.00),
('ORD005', '2023-06-05', '2023-06-09', 'Ekspress', 'CUST005', 'Michael Wilson', 'Rumah', 'Indonesia', 'Yogyakarta', 'DI Yogyakarta', '13579', 'Wilayah B', 'PROD002', 'Pakaian', 'Celana', 'Celana Jeans', 150.00, 2, 15.00, 50.00),
('ORD006', '2023-06-06', '2023-06-10', 'Reguler', 'CUST006', 'Emily Taylor', 'Kantor', 'Indonesia', 'Semarang', 'Jawa Tengah', '86420', 'Wilayah C', 'PROD003', 'Makanan', 'Minuman', 'Air Mineral', 1.00, 10, 0.50, 5.00),
('ORD007', '2023-06-07', '2023-06-11', 'Reguler', 'CUST007', 'Daniel Anderson', 'Rumah', 'Indonesia', 'Makassar', 'Sulawesi Selatan', '24680', 'Wilayah A', 'PROD001', 'Elektronik', 'Televisi', 'TV LED 55"', 1200.00, 1, 50.00, 300.00),
('ORD008', '2023-06-08', '2023-06-12', 'Ekspress', 'CUST008', 'Olivia Martinez', 'Kantor', 'Indonesia', 'Palembang', 'Sumatera Selatan', '80246', 'Wilayah B', 'PROD002', 'Pakaian', 'Dress', 'Dress Merah', 200.00, 3, 10.00, 80.00),
('ORD009', '2023-06-09', '2023-06-13', 'Reguler', 'CUST009', 'James Lee', 'Rumah', 'Indonesia', 'Balikpapan', 'Kalimantan Timur', '57904', 'Wilayah C', 'PROD003', 'Makanan', 'Minuman', 'Soda', 2.50, 5, 0.50, 5.00),
('ORD010', '2023-06-10', '2023-06-14', 'Reguler', 'CUST010', 'Sophia Garcia', 'Kantor', 'Indonesia', 'Denpasar', 'Bali', '90210', 'Wilayah A', 'PROD001', 'Elektronik', 'Handphone', 'Smartphone', 1000.00, 2, 50.00, 400.00);

SELECT * FROM tokopaedi