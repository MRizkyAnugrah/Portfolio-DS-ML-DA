SELECT customer_id, customer_name, AVG(sales) AS average_spending
FROM orders
GROUP BY customer_id, customer_name
HAVING AVG(sales) > (SELECT AVG(sales) FROM orders)
ORDER BY AVG(sales) DESC