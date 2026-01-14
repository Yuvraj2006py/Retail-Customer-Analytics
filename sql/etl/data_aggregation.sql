-- Pre-Aggregated Tables for Dashboard Performance
-- These tables can be refreshed periodically to improve dashboard query performance

-- Daily Transaction Summary
CREATE TABLE IF NOT EXISTS agg_daily_transactions AS
SELECT 
    CAST(TransactionDate AS DATE) AS TransactionDate,
    COUNT(DISTINCT TransactionID) AS TransactionCount,
    COUNT(DISTINCT CustomerID) AS UniqueCustomers,
    COUNT(DISTINCT ProductID) AS UniqueProducts,
    SUM(Quantity) AS TotalQuantity,
    SUM(LineTotal) AS TotalRevenue,
    AVG(LineTotal) AS AvgTransactionValue
FROM fact_transactions
GROUP BY CAST(TransactionDate AS DATE);

-- Customer Segment Summary
CREATE TABLE IF NOT EXISTS agg_customer_segments AS
WITH CustomerRFM AS (
    SELECT 
        CustomerID,
        DATEDIFF(DAY, MAX(TransactionDate), GETDATE()) AS Recency,
        COUNT(DISTINCT TransactionDate) AS Frequency,
        SUM(LineTotal) AS Monetary
    FROM fact_transactions
    GROUP BY CustomerID
),
RFMSegments AS (
    SELECT 
        CustomerID,
        Recency,
        Frequency,
        Monetary,
        CASE 
            WHEN Recency <= 30 AND Frequency >= 20 AND Monetary >= 5000 THEN 'Champions'
            WHEN Recency <= 60 AND Frequency >= 10 AND Monetary >= 2000 THEN 'Loyal Customers'
            WHEN Recency <= 90 AND Frequency >= 5 THEN 'Potential Loyalists'
            WHEN Recency <= 30 AND Frequency <= 5 THEN 'New Customers'
            WHEN Recency > 90 AND Frequency >= 10 THEN 'At Risk'
            WHEN Recency > 180 THEN 'Lost'
            ELSE 'Need Attention'
        END AS Segment
    FROM CustomerRFM
)
SELECT 
    Segment,
    COUNT(*) AS CustomerCount,
    AVG(Recency) AS AvgRecency,
    AVG(Frequency) AS AvgFrequency,
    AVG(Monetary) AS AvgMonetary
FROM RFMSegments
GROUP BY Segment;

-- Product Category Performance Summary
CREATE TABLE IF NOT EXISTS agg_category_performance AS
SELECT 
    p.Category,
    COUNT(DISTINCT t.TransactionID) AS TransactionCount,
    COUNT(DISTINCT t.CustomerID) AS UniqueCustomers,
    SUM(t.Quantity) AS TotalUnitsSold,
    SUM(t.LineTotal) AS TotalRevenue,
    AVG(t.LineTotal) AS AvgTransactionValue,
    COUNT(DISTINCT p.ProductID) AS ProductCount
FROM fact_transactions t
INNER JOIN dim_products p ON t.ProductID = p.ProductID
GROUP BY p.Category;

-- Monthly Loyalty Summary
CREATE TABLE IF NOT EXISTS agg_monthly_loyalty AS
SELECT 
    YEAR(TransactionDate) AS Year,
    MONTH(TransactionDate) AS Month,
    COUNT(DISTINCT CustomerID) AS ActiveMembers,
    SUM(PointsEarned) AS TotalPointsEarned,
    SUM(PointsRedeemed) AS TotalPointsRedeemed,
    AVG(PointsBalance) AS AvgPointsBalance,
    COUNT(DISTINCT CASE WHEN Tier = 'Platinum' THEN CustomerID END) AS PlatinumCount,
    COUNT(DISTINCT CASE WHEN Tier = 'Gold' THEN CustomerID END) AS GoldCount,
    COUNT(DISTINCT CASE WHEN Tier = 'Silver' THEN CustomerID END) AS SilverCount,
    COUNT(DISTINCT CASE WHEN Tier = 'Bronze' THEN CustomerID END) AS BronzeCount
FROM fact_loyalty_points
GROUP BY YEAR(TransactionDate), MONTH(TransactionDate);
