-- Loyalty Program Analysis Queries
-- PC Optimum loyalty program metrics

-- Loyalty Program Overview
SELECT 
    COUNT(DISTINCT CustomerID) AS TotalMembers,
    SUM(PointsEarned) AS TotalPointsEarned,
    SUM(PointsRedeemed) AS TotalPointsRedeemed,
    AVG(PointsBalance) AS AvgPointsBalance,
    COUNT(DISTINCT CASE WHEN Tier = 'Platinum' THEN CustomerID END) AS PlatinumMembers,
    COUNT(DISTINCT CASE WHEN Tier = 'Gold' THEN CustomerID END) AS GoldMembers,
    COUNT(DISTINCT CASE WHEN Tier = 'Silver' THEN CustomerID END) AS SilverMembers,
    COUNT(DISTINCT CASE WHEN Tier = 'Bronze' THEN CustomerID END) AS BronzeMembers
FROM fact_loyalty_points
INNER JOIN (
    SELECT CustomerID, MAX(TransactionDate) AS MaxDate
    FROM fact_loyalty_points
    GROUP BY CustomerID
) latest ON fact_loyalty_points.CustomerID = latest.CustomerID 
    AND fact_loyalty_points.TransactionDate = latest.MaxDate;

-- Points Activity Over Time
SELECT 
    YEAR(TransactionDate) AS Year,
    MONTH(TransactionDate) AS Month,
    SUM(PointsEarned) AS PointsEarned,
    SUM(PointsRedeemed) AS PointsRedeemed,
    AVG(PointsBalance) AS AvgPointsBalance,
    COUNT(DISTINCT CustomerID) AS ActiveMembers
FROM fact_loyalty_points
GROUP BY YEAR(TransactionDate), MONTH(TransactionDate)
ORDER BY Year DESC, Month DESC;

-- Tier Migration Analysis
WITH CustomerTierHistory AS (
    SELECT 
        CustomerID,
        TransactionDate,
        Tier,
        LAG(Tier) OVER (PARTITION BY CustomerID ORDER BY TransactionDate) AS PreviousTier
    FROM fact_loyalty_points
)
SELECT 
    PreviousTier AS FromTier,
    Tier AS ToTier,
    COUNT(*) AS MigrationCount
FROM CustomerTierHistory
WHERE PreviousTier IS NOT NULL AND PreviousTier != Tier
GROUP BY PreviousTier, Tier
ORDER BY MigrationCount DESC;

-- Redemption Patterns
SELECT 
    CASE 
        WHEN PointsRedeemed BETWEEN 0 AND 1000 THEN '0-1K'
        WHEN PointsRedeemed BETWEEN 1001 AND 5000 THEN '1K-5K'
        WHEN PointsRedeemed BETWEEN 5001 AND 10000 THEN '5K-10K'
        ELSE '10K+'
    END AS RedemptionRange,
    COUNT(*) AS RedemptionCount,
    AVG(PointsRedeemed) AS AvgRedemption,
    SUM(PointsRedeemed) AS TotalRedeemed
FROM fact_loyalty_points
WHERE PointsRedeemed > 0
GROUP BY 
    CASE 
        WHEN PointsRedeemed BETWEEN 0 AND 1000 THEN '0-1K'
        WHEN PointsRedeemed BETWEEN 1001 AND 5000 THEN '1K-5K'
        WHEN PointsRedeemed BETWEEN 5001 AND 10000 THEN '5K-10K'
        ELSE '10K+'
    END
ORDER BY RedemptionRange;
