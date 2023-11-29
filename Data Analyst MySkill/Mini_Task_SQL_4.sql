SELECT product_name
FROM orders
GROUP BY product_name
ORDER BY SUM(quantity) DESC
LIMIT 1

SELECT customer_id, customer_name
FROM orders
WHERE segment = 'Consumer'
GROUP BY customer_id, customer_name
ORDER BY AVG(sales) DESC
LIMIT 100

SELECT product_name, quantity
FROM orders
WHERE customer_name = 'Joel Eaton'

SELECT order_date
FROM orders
GROUP BY order_date
ORDER BY COUNT(order_id) DESC
LIMIT 1

SELECT product_name
FROM orders
GROUP BY product_name
ORDER BY AVG(profit / quantity) DESC
LIMIT 1