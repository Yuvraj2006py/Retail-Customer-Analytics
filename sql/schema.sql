-- Retail Analytics Database Schema
-- Star Schema Design for Data Warehouse

-- Dimension Tables

-- Customer Dimension
CREATE TABLE IF NOT EXISTS dim_customers (
    CustomerID VARCHAR(50) PRIMARY KEY,
    FirstName VARCHAR(100),
    LastName VARCHAR(100),
    Email VARCHAR(255),
    Phone VARCHAR(50),
    Address TEXT,
    City VARCHAR(100),
    Province VARCHAR(10),
    PostalCode VARCHAR(20),
    EnrollmentDate DATE,
    DateOfBirth DATE
);

-- Product Dimension
CREATE TABLE IF NOT EXISTS dim_products (
    ProductID VARCHAR(50) PRIMARY KEY,
    ProductName VARCHAR(255),
    Category VARCHAR(100),
    Price DECIMAL(10, 2),
    Cost DECIMAL(10, 2)
);

-- Store Dimension
CREATE TABLE IF NOT EXISTS dim_stores (
    StoreID VARCHAR(50) PRIMARY KEY,
    StoreName VARCHAR(255),
    City VARCHAR(100),
    Province VARCHAR(10),
    Address TEXT
);

-- Date Dimension
CREATE TABLE IF NOT EXISTS dim_dates (
    DateKey INT PRIMARY KEY,
    Date DATE NOT NULL,
    Year INT,
    Quarter INT,
    Month INT,
    MonthName VARCHAR(20),
    Week INT,
    DayOfWeek INT,
    DayName VARCHAR(20),
    DayOfMonth INT,
    IsWeekend BOOLEAN,
    IsHolidaySeason BOOLEAN
);

-- Fact Tables

-- Transactions Fact Table
CREATE TABLE IF NOT EXISTS fact_transactions (
    TransactionID VARCHAR(50),
    CustomerID VARCHAR(50),
    TransactionDate DATE,
    StoreID VARCHAR(50),
    ProductID VARCHAR(50),
    Quantity INT,
    UnitPrice DECIMAL(10, 2),
    LineTotal DECIMAL(10, 2),
    PRIMARY KEY (TransactionID, CustomerID, ProductID, TransactionDate),
    FOREIGN KEY (CustomerID) REFERENCES dim_customers(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES dim_products(ProductID),
    FOREIGN KEY (StoreID) REFERENCES dim_stores(StoreID)
);

-- Loyalty Points Fact Table
CREATE TABLE IF NOT EXISTS fact_loyalty_points (
    LoyaltyRecordID INT PRIMARY KEY,
    CustomerID VARCHAR(50),
    TransactionID VARCHAR(50),
    TransactionDate DATE,
    PointsEarned INT,
    PointsRedeemed INT,
    PointsBalance INT,
    Tier VARCHAR(20),
    PointsPerDollar DECIMAL(5, 2),
    FOREIGN KEY (CustomerID) REFERENCES dim_customers(CustomerID),
    FOREIGN KEY (TransactionID) REFERENCES fact_transactions(TransactionID)
);

-- Surveys Fact Table
CREATE TABLE IF NOT EXISTS fact_surveys (
    SurveyID INT PRIMARY KEY,
    CustomerID VARCHAR(50),
    SurveyDate DATE,
    SatisfactionScore INT CHECK (SatisfactionScore BETWEEN 1 AND 5),
    NPSScore INT CHECK (NPSScore BETWEEN 0 AND 10),
    Feedback TEXT,
    WouldRecommend VARCHAR(10),
    FOREIGN KEY (CustomerID) REFERENCES dim_customers(CustomerID)
);

-- Indexes for Performance

-- Transaction indexes
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON fact_transactions(CustomerID);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON fact_transactions(TransactionDate);
CREATE INDEX IF NOT EXISTS idx_transactions_product ON fact_transactions(ProductID);
CREATE INDEX IF NOT EXISTS idx_transactions_store ON fact_transactions(StoreID);

-- Loyalty indexes
CREATE INDEX IF NOT EXISTS idx_loyalty_customer ON fact_loyalty_points(CustomerID);
CREATE INDEX IF NOT EXISTS idx_loyalty_date ON fact_loyalty_points(TransactionDate);
CREATE INDEX IF NOT EXISTS idx_loyalty_tier ON fact_loyalty_points(Tier);

-- Survey indexes
CREATE INDEX IF NOT EXISTS idx_surveys_customer ON fact_surveys(CustomerID);
CREATE INDEX IF NOT EXISTS idx_surveys_date ON fact_surveys(SurveyDate);
