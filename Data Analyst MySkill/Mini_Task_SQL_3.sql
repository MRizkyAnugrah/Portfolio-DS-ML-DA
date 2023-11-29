SELECT DISTINCT customer_name
FROM orders
WHERE segment = 'Consumer' AND city = 'San Francisco' AND EXTRACT(MONTH FROM order_date) = 5 AND EXTRACT(YEAR FROM order_date) = 2014;

SELECT *
FROM orders
ORDER BY profit DESC
LIMIT 10;

SELECT *
FROM orders
ORDER BY sales DESC
LIMIT 10;
