SELECT * FROM order_detail

SELECT * FROM sku_detail

SELECT * FROM customer_detail

SELECT * FROM payment_detail

-- Nomor 1

SELECT EXTRACT(MONTH FROM order_date) AS month, SUM(after_discount) AS total_value
FROM order_detail
WHERE is_valid = 1 AND EXTRACT(YEAR FROM order_date) = 2021
GROUP BY month
ORDER BY total_value DESC
LIMIT 1;

-- Nomor 2

SELECT EXTRACT(MONTH FROM order_date) AS month, 
       COUNT(DISTINCT customer_id) AS total_customers, 
       COUNT(DISTINCT id) AS total_orders, 
       SUM(qty_ordered) AS total_quantity
FROM order_detail
WHERE is_valid = 1 AND EXTRACT(YEAR FROM order_date) = 2021
GROUP BY month
ORDER BY total_quantity DESC
LIMIT 1;

-- Nomor 3

SELECT sku_detail.category, SUM(order_detail.after_discount) AS total_value
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
WHERE order_detail.is_valid = 1 AND EXTRACT(YEAR FROM order_detail.order_date) = 2022
GROUP BY sku_detail.category
ORDER BY total_value DESC
LIMIT 1;

-- Nomor 4

WITH category_values AS (
  SELECT sku_detail.category,
         SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 THEN order_detail.after_discount ELSE 0 END) AS total_value_2021,
         SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2022 THEN order_detail.after_discount ELSE 0 END) AS total_value_2022
  FROM order_detail
  JOIN sku_detail ON order_detail.sku_id = sku_detail.id
  WHERE order_detail.is_valid = 1
  GROUP BY sku_detail.category
)
SELECT category, 
       total_value_2021,
       total_value_2022,
       CASE WHEN total_value_2022 > total_value_2021 THEN 'Peningkatan'
            WHEN total_value_2022 < total_value_2021 THEN 'Penurunan'
            ELSE 'Tidak berubah'
       END AS nilai_transaksi
FROM category_values
WHERE total_value_2021 != total_value_2022;

-- Nomor 5

SELECT sku_detail.sku_name,
       sku_detail.category,
       COUNT(DISTINCT order_detail.customer_id) AS total_customers,
       COUNT(DISTINCT order_detail.id) AS total_orders,
       SUM(order_detail.qty_ordered) AS total_quantity
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
WHERE EXTRACT(YEAR FROM order_detail.order_date) = 2022
  AND order_detail.is_valid = 1
GROUP BY sku_detail.sku_name, sku_detail.category
ORDER BY SUM(order_detail.after_discount) DESC
LIMIT 10;

-- Nomor 6

SELECT payment_detail.payment_method,
       COUNT(DISTINCT order_detail.id) AS total_unique_orders
FROM order_detail
JOIN payment_detail ON order_detail.payment_id = payment_detail.id
WHERE EXTRACT(YEAR FROM order_detail.order_date) = 2022
  AND order_detail.is_valid = 1
GROUP BY payment_detail.payment_method
ORDER BY total_unique_orders DESC
LIMIT 5;

-- Nomor 7

SELECT 
  CASE
    WHEN lower(sku_detail.sku_name) LIKE '%samsung%' THEN 'Samsung'
    WHEN lower(sku_detail.sku_name) LIKE '%apple%' THEN 'Apple'
    WHEN lower(sku_detail.sku_name) LIKE '%sony%' THEN 'Sony'
    WHEN lower(sku_detail.sku_name) LIKE '%huawei%' THEN 'Huawei'
    WHEN lower(sku_detail.sku_name) LIKE '%lenovo%' THEN 'Lenovo'
  END AS product,
  SUM(order_detail.after_discount) AS total_transaction_value
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
WHERE order_detail.is_valid = 1
  AND (lower(sku_detail.sku_name) LIKE '%samsung%'
       OR lower(sku_detail.sku_name) LIKE '%apple%'
       OR lower(sku_detail.sku_name) LIKE '%sony%'
       OR lower(sku_detail.sku_name) LIKE '%huawei%'
       OR lower(sku_detail.sku_name) LIKE '%lenovo%')
GROUP BY product
ORDER BY total_transaction_value DESC;

-- Nomor 8

SELECT
  sku_detail.category,
  SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 
	  THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END) AS profit_2021,
  SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2022 
	  THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END) AS profit_2022,
  ((SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2022 
		THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END) - 
	SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 THEN order_detail.after_discount - 
		(sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END)) / 
   SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 THEN order_detail.after_discount - 
	   (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END)) * 100 AS profit_difference_percentage
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
WHERE order_detail.is_valid = 1
  AND (EXTRACT(YEAR FROM order_detail.order_date) = 2021 OR EXTRACT(YEAR FROM order_detail.order_date) = 2022)
GROUP BY sku_detail.category
ORDER BY profit_difference_percentage DESC;

-- Nomor 9

WITH profit_data AS (
  SELECT
    sku_detail.category,
    sku_detail.id AS sku_id,
    SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 
		THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END) AS profit_2021,
    SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2022 
		THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END) AS profit_2022,
    (CASE
      WHEN SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 
			   THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END) = 0 THEN NULL
      ELSE ((SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2022 
				 THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END) - 
			 SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 
				 THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END)) / 
			SUM(CASE WHEN EXTRACT(YEAR FROM order_detail.order_date) = 2021 
				THEN order_detail.after_discount - (sku_detail.cogs * order_detail.qty_ordered) ELSE 0 END)) * 100
    END) AS profit_difference_percentage
  FROM order_detail
  JOIN sku_detail ON order_detail.sku_id = sku_detail.id
  WHERE order_detail.is_valid = 1
    AND (EXTRACT(YEAR FROM order_detail.order_date) = 2021 OR EXTRACT(YEAR FROM order_detail.order_date) = 2022)
  GROUP BY sku_detail.category, sku_detail.id
)
SELECT
  sku_detail.sku_name,
  sku_detail.category,
  profit_data.profit_2021,
  profit_data.profit_2022,
  profit_data.profit_difference_percentage
FROM profit_data
JOIN sku_detail ON profit_data.sku_id = sku_detail.id
WHERE profit_data.profit_difference_percentage > 0
ORDER BY profit_difference_percentage DESC
LIMIT 5;

-- Nomor 10

SELECT sku_detail.category, COUNT(DISTINCT order_detail.id) AS total_unique_orders
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
JOIN payment_detail ON order_detail.payment_id = payment_detail.id
WHERE payment_detail.payment_method = 'cod'
  AND order_detail.is_valid = 1
  AND EXTRACT(YEAR FROM order_detail.order_date) = 2022
GROUP BY sku_detail.category
ORDER BY total_unique_orders DESC;

SELECT sku_detail.category, COUNT(DISTINCT order_detail.id) AS total_unique_orders
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
JOIN payment_detail ON order_detail.payment_id = payment_detail.id
WHERE payment_detail.payment_method = 'Payaxis'
  AND order_detail.is_valid = 1
  AND EXTRACT(YEAR FROM order_detail.order_date) = 2022
GROUP BY sku_detail.category
ORDER BY total_unique_orders DESC;

SELECT sku_detail.category, COUNT(DISTINCT order_detail.id) AS total_unique_orders
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
JOIN payment_detail ON order_detail.payment_id = payment_detail.id
WHERE payment_detail.payment_method = 'Easypay'
  AND order_detail.is_valid = 1
  AND EXTRACT(YEAR FROM order_detail.order_date) = 2022
GROUP BY sku_detail.category
ORDER BY total_unique_orders DESC;

SELECT sku_detail.category, COUNT(DISTINCT order_detail.id) AS total_unique_orders
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
JOIN payment_detail ON order_detail.payment_id = payment_detail.id
WHERE payment_detail.payment_method = 'customercredit'
  AND order_detail.is_valid = 1
  AND EXTRACT(YEAR FROM order_detail.order_date) = 2022
GROUP BY sku_detail.category
ORDER BY total_unique_orders DESC;

SELECT sku_detail.category, COUNT(DISTINCT order_detail.id) AS total_unique_orders
FROM order_detail
JOIN sku_detail ON order_detail.sku_id = sku_detail.id
JOIN payment_detail ON order_detail.payment_id = payment_detail.id
WHERE payment_detail.payment_method = 'jazzwallet'
  AND order_detail.is_valid = 1
  AND EXTRACT(YEAR FROM order_detail.order_date) = 2022
GROUP BY sku_detail.category
ORDER BY total_unique_orders DESC;