-- Product Performance Queries
-- Product sales and revenue analysis

-- Top Products by Revenue
SELECT 
    p.ProductID,
    p.ProductName,
    p.Category,
    COUNT(DISTINCT t.TransactionID) AS TransactionCount,
    SUM(t.Quantity) AS TotalUnitsSold,
    SUM(t.LineTotal) AS TotalRevenue,
    AVG(t.UnitPrice) AS AvgPrice,
    AVG(t.Quantity) AS AvgQuantityPerTransaction
FROM dim_products p
INNER JOIN fact_transactions t ON p.ProductID = t.ProductID
GROUP BY p.ProductID, p.ProductName, p.Category
ORDER BY TotalRevenue DESC;

-- Product Performance by Category
SELECT 
    p.Category,
    COUNT(DISTINCT p.ProductID) AS ProductCount,
    COUNT(DISTINCT t.TransactionID) AS TransactionCount,
    SUM(t.Quantity) AS TotalUnitsSold,
    SUM(t.LineTotal) AS TotalRevenue,
    AVG(t.LineTotal) AS AvgTransactionValue
FROM dim_products p
INNER JOIN fact_transactions t ON p.ProductID = t.ProductID
GROUP BY p.Category
ORDER BY TotalRevenue DESC;

-- Product Sales Trends (Monthly)
SELECT 
    YEAR(t.TransactionDate) AS Year,
    MONTH(t.TransactionDate) AS Month,
    p.ProductID,
    p.ProductName,
    p.Category,
    SUM(t.Quantity) AS UnitsSold,
    SUM(t.LineTotal) AS Revenue
FROM fact_transactions t
INNER JOIN dim_products p ON t.ProductID = p.ProductID
GROUP BY YEAR(t.TransactionDate), MONTH(t.TransactionDate), 
         p.ProductID, p.ProductName, p.Category
ORDER BY Year DESC, Month DESC, Revenue DESC;

-- Cross-Sell Opportunities (Products frequently bought together)
WITH ProductPairs AS (
    SELECT 
        t1.ProductID AS Product1,
        t2.ProductID AS Product2,
        COUNT(*) AS CoOccurrence
    FROM fact_transactions t1
    INNER JOIN fact_transactions t2 
        ON t1.TransactionID = t2.TransactionID
        AND t1.ProductID < t2.ProductID
    GROUP BY t1.ProductID, t2.ProductID
)
SELECT 
    p1.ProductName AS Product1,
    p2.ProductName AS Product2,
    pp.CoOccurrence
FROM ProductPairs pp
INNER JOIN dim_products p1 ON pp.Product1 = p1.ProductID
INNER JOIN dim_products p2 ON pp.Product2 = p2.ProductID
WHERE pp.CoOccurrence >= 10  -- Minimum threshold
ORDER BY pp.CoOccurrence DESC;
