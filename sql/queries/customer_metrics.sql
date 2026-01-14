-- Customer Metrics Queries
-- Aggregated customer-level metrics for analytics

-- Customer Lifetime Value (CLV)
SELECT 
    c.CustomerID,
    c.FirstName,
    c.LastName,
    c.EnrollmentDate,
    COUNT(DISTINCT t.TransactionDate) AS TotalTransactions,
    SUM(t.LineTotal) AS TotalSpent,
    AVG(t.LineTotal) AS AvgTransactionValue,
    MAX(t.TransactionDate) AS LastPurchaseDate,
    MIN(t.TransactionDate) AS FirstPurchaseDate,
    DATEDIFF(DAY, MAX(t.TransactionDate), GETDATE()) AS DaysSinceLastPurchase,
    DATEDIFF(DAY, MIN(t.TransactionDate), MAX(t.TransactionDate)) AS CustomerLifespanDays
FROM dim_customers c
LEFT JOIN fact_transactions t ON c.CustomerID = t.CustomerID
GROUP BY c.CustomerID, c.FirstName, c.LastName, c.EnrollmentDate;

-- Customer RFM Scores
WITH CustomerMetrics AS (
    SELECT 
        CustomerID,
        MAX(TransactionDate) AS LastPurchaseDate,
        COUNT(DISTINCT TransactionDate) AS Frequency,
        SUM(LineTotal) AS Monetary
    FROM fact_transactions
    GROUP BY CustomerID
),
RFMScores AS (
    SELECT 
        CustomerID,
        DATEDIFF(DAY, LastPurchaseDate, GETDATE()) AS Recency,
        Frequency,
        Monetary,
        NTILE(5) OVER (ORDER BY DATEDIFF(DAY, LastPurchaseDate, GETDATE()) DESC) AS RecencyScore,
        NTILE(5) OVER (ORDER BY Frequency) AS FrequencyScore,
        NTILE(5) OVER (ORDER BY Monetary) AS MonetaryScore
    FROM CustomerMetrics
)
SELECT 
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    RecencyScore,
    FrequencyScore,
    MonetaryScore,
    (RecencyScore * 100 + FrequencyScore * 10 + MonetaryScore) AS RFMScore
FROM RFMScores;

-- Customer Loyalty Tier Distribution
SELECT 
    l.Tier,
    COUNT(DISTINCT l.CustomerID) AS CustomerCount,
    AVG(l.PointsBalance) AS AvgPointsBalance,
    SUM(l.PointsEarned) AS TotalPointsEarned,
    SUM(l.PointsRedeemed) AS TotalPointsRedeemed
FROM fact_loyalty_points l
INNER JOIN (
    SELECT CustomerID, MAX(TransactionDate) AS MaxDate
    FROM fact_loyalty_points
    GROUP BY CustomerID
) latest ON l.CustomerID = latest.CustomerID 
    AND l.TransactionDate = latest.MaxDate
GROUP BY l.Tier
ORDER BY 
    CASE l.Tier
        WHEN 'Platinum' THEN 4
        WHEN 'Gold' THEN 3
        WHEN 'Silver' THEN 2
        WHEN 'Bronze' THEN 1
    END DESC;
