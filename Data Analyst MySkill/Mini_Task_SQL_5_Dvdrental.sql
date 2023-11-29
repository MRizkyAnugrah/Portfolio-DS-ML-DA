CREATE TABLE rental_history AS
SELECT r.rental_id, r.rental_date, r.return_date, f.title AS film_title, c.first_name || ' ' || c.last_name AS customer_name
FROM rental r
JOIN customer c ON r.customer_id = c.customer_id
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN film f ON i.film_id = f.film_id

SELECT * FROM rental_history