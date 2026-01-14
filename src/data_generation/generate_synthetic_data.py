"""
Synthetic Data Generation for Retail Customer Engagement & Loyalty Analytics

This module generates realistic synthetic datasets for:
1. Transactional data
2. PC Optimum loyalty program data
3. Customer survey data

All datasets are linked by CustomerID to ensure realistic relationships.
"""

import pandas as pd
import numpy as np
from faker import Faker
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import random


class SyntheticDataGenerator:
    """Generate synthetic retail customer data."""
    
    def __init__(self, config_path: str = "config/data_generation_config.yaml"):
        """
        Initialize the data generator.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.fake = Faker('en_CA')  # Canadian locale for PC Optimum
        Faker.seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Extract config values
        self.num_customers = self.config['data_generation']['num_customers']
        self.start_date = datetime.strptime(
            self.config['data_generation']['start_date'], '%Y-%m-%d'
        )
        self.end_date = datetime.strptime(
            self.config['data_generation']['end_date'], '%Y-%m-%d'
        )
        
        # Initialize data structures
        self.customers = None
        self.products = None
        self.stores = None
        self.transactions_df = None
        self.loyalty_df = None
        self.survey_df = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_products(self) -> pd.DataFrame:
        """
        Generate product catalog.
        
        Returns:
            DataFrame with product information
        """
        num_products = self.config['data_generation']['products']['num_products']
        num_categories = self.config['data_generation']['products']['num_categories']
        min_price = self.config['data_generation']['products']['min_price']
        max_price = self.config['data_generation']['products']['max_price']
        
        categories = [
            'Electronics', 'Clothing', 'Food & Beverages', 'Home & Garden',
            'Health & Beauty', 'Sports & Outdoors', 'Toys & Games',
            'Books & Media', 'Automotive', 'Pet Supplies', 'Office Supplies',
            'Baby Products', 'Furniture', 'Jewelry', 'Tools & Hardware'
        ][:num_categories]
        
        products = []
        for i in range(1, num_products + 1):
            category = np.random.choice(categories)
            price = round(np.random.uniform(min_price, max_price), 2)
            
            products.append({
                'ProductID': f'PROD_{i:05d}',
                'ProductName': self.fake.catch_phrase(),
                'Category': category,
                'Price': price,
                'Cost': round(price * np.random.uniform(0.4, 0.7), 2)  # Cost is 40-70% of price
            })
        
        self.products = pd.DataFrame(products)
        return self.products
    
    def generate_stores(self) -> pd.DataFrame:
        """
        Generate store locations.
        
        Returns:
            DataFrame with store information
        """
        num_stores = self.config['data_generation']['stores']['num_stores']
        cities = self.config['data_generation']['stores']['store_cities']
        
        stores = []
        for i in range(1, num_stores + 1):
            city = np.random.choice(cities)
            stores.append({
                'StoreID': f'STORE_{i:03d}',
                'StoreName': f'{city} Store #{i}',
                'City': city,
                'Province': self._get_province(city),
                'Address': self.fake.address()
            })
        
        self.stores = pd.DataFrame(stores)
        return self.stores
    
    def _get_province(self, city: str) -> str:
        """Map city to province."""
        city_province = {
            'Toronto': 'ON', 'Ottawa': 'ON',
            'Vancouver': 'BC',
            'Montreal': 'QC', 'Quebec City': 'QC',
            'Calgary': 'AB', 'Edmonton': 'AB',
            'Winnipeg': 'MB'
        }
        return city_province.get(city, 'ON')
    
    def generate_customers(self) -> pd.DataFrame:
        """
        Generate customer base.
        
        Returns:
            DataFrame with customer information
        """
        customers = []
        for i in range(1, self.num_customers + 1):
            enrollment_start = datetime.strptime(
                self.config['data_generation']['loyalty']['enrollment_start_date'], '%Y-%m-%d'
            ).date()
            enrollment_end = datetime.strptime(
                self.config['data_generation']['loyalty']['enrollment_end_date'], '%Y-%m-%d'
            ).date()
            enrollment_date = self.fake.date_between(
                start_date=enrollment_start,
                end_date=enrollment_end
            )
            
            customers.append({
                'CustomerID': f'CUST_{i:05d}',
                'FirstName': self.fake.first_name(),
                'LastName': self.fake.last_name(),
                'Email': self.fake.email(),
                'Phone': self.fake.phone_number(),
                'Address': self.fake.address(),
                'City': self.fake.city(),
                'Province': self.fake.province_abbr(),
                'PostalCode': self.fake.postalcode(),
                'EnrollmentDate': enrollment_date,
                'DateOfBirth': self.fake.date_of_birth(minimum_age=18, maximum_age=80)
            })
        
        self.customers = pd.DataFrame(customers)
        return self.customers
    
    def generate_transactions(self) -> pd.DataFrame:
        """
        Generate transactional data with realistic patterns.
        
        Returns:
            DataFrame with transaction information
        """
        transactions = []
        transaction_id = 1
        
        # Create lookup dictionaries for O(1) access (OPTIMIZATION)
        customer_enrollment = dict(zip(
            self.customers['CustomerID'],
            pd.to_datetime(self.customers['EnrollmentDate'])
        ))
        product_prices = dict(zip(
            self.products['ProductID'],
            self.products['Price']
        ))
        store_ids = self.stores['StoreID'].values
        product_ids = self.products['ProductID'].values
        
        # Customer segments for realistic behavior
        high_value_pct = 0.15  # 15% high-value customers
        loyal_pct = 0.25  # 25% loyal customers
        occasional_pct = 0.40  # 40% occasional customers
        new_pct = 0.20  # 20% new customers
        
        customer_segments = {}
        for customer_id in self.customers['CustomerID']:
            rand = np.random.random()
            if rand < high_value_pct:
                customer_segments[customer_id] = 'high_value'
            elif rand < high_value_pct + loyal_pct:
                customer_segments[customer_id] = 'loyal'
            elif rand < high_value_pct + loyal_pct + occasional_pct:
                customer_segments[customer_id] = 'occasional'
            else:
                customer_segments[customer_id] = 'new'
        
        for customer_id in self.customers['CustomerID']:
            segment = customer_segments[customer_id]
            enrollment_date = customer_enrollment[customer_id]  # OPTIMIZED: O(1) lookup
            
            # Determine transaction frequency based on segment
            if segment == 'high_value':
                num_transactions = np.random.randint(80, 200)
                days_active = (self.end_date - enrollment_date).days
                avg_days_between = max(1, days_active // num_transactions)
            elif segment == 'loyal':
                num_transactions = np.random.randint(40, 100)
                days_active = (self.end_date - enrollment_date).days
                avg_days_between = max(2, days_active // num_transactions)
            elif segment == 'occasional':
                num_transactions = np.random.randint(5, 30)
                days_active = (self.end_date - enrollment_date).days
                avg_days_between = max(10, days_active // num_transactions)
            else:  # new
                num_transactions = np.random.randint(1, 10)
                days_active = (self.end_date - enrollment_date).days
                avg_days_between = max(20, days_active // num_transactions) if days_active > 0 else 30
            
            # Generate transactions
            current_date = max(enrollment_date, self.start_date)
            transaction_dates = []
            
            for _ in range(num_transactions):
                if current_date > self.end_date:
                    break
                
                # Add some randomness to transaction dates
                days_to_add = np.random.poisson(avg_days_between)
                current_date += timedelta(days=days_to_add)
                
                if current_date <= self.end_date:
                    transaction_dates.append(current_date)
            
            # Generate transaction details
            for trans_date in transaction_dates:
                store_id = np.random.choice(store_ids)  # OPTIMIZED: Use pre-extracted array
                
                # Number of items in transaction (basket size)
                if segment == 'high_value':
                    num_items = np.random.randint(3, 15)
                elif segment == 'loyal':
                    num_items = np.random.randint(2, 10)
                else:
                    num_items = np.random.randint(1, 6)
                
                # Select products
                selected_products = np.random.choice(
                    product_ids,  # OPTIMIZED: Use pre-extracted array
                    size=num_items,
                    replace=True
                )
                
                for product_id in selected_products:
                    price = product_prices[product_id]  # OPTIMIZED: O(1) lookup instead of DataFrame filter
                    quantity = np.random.randint(1, 5)
                    line_total = round(price * quantity, 2)
                    
                    transactions.append({
                        'TransactionID': f'TXN_{transaction_id:06d}',
                        'CustomerID': customer_id,
                        'TransactionDate': trans_date,
                        'StoreID': store_id,
                        'ProductID': product_id,
                        'Quantity': quantity,
                        'UnitPrice': price,
                        'LineTotal': line_total
                    })
                
                transaction_id += 1
        
        self.transactions_df = pd.DataFrame(transactions)
        
        # Add seasonality (more transactions during holidays)
        self._add_seasonality()
        
        return self.transactions_df
    
    def _add_seasonality(self):
        """Add seasonal patterns to transactions."""
        # Boost transactions during holiday seasons
        holiday_months = [11, 12, 1, 2]  # Nov, Dec, Jan, Feb
        
        # OPTIMIZED: Use vectorized operations instead of iterrows()
        if len(self.transactions_df) > 0:
            transaction_dates = pd.to_datetime(self.transactions_df['TransactionDate'])
            months = transaction_dates.dt.month
            is_holiday = months.isin(holiday_months)
            
            # Apply seasonal boost (30% chance for holiday transactions)
            random_mask = np.random.random(len(self.transactions_df)) < 0.3
            boost_mask = is_holiday & random_mask
            
            if boost_mask.any():
                # Vectorized update
                self.transactions_df.loc[boost_mask, 'Quantity'] = (
                    self.transactions_df.loc[boost_mask, 'Quantity'] + 1
                ).clip(upper=10)
                
                # Recalculate LineTotal
                self.transactions_df.loc[boost_mask, 'LineTotal'] = (
                    self.transactions_df.loc[boost_mask, 'UnitPrice'] * 
                    self.transactions_df.loc[boost_mask, 'Quantity']
                ).round(2)
    
    def generate_loyalty_data(self) -> pd.DataFrame:
        """
        Generate PC Optimum loyalty program data.
        
        Returns:
            DataFrame with loyalty program information
        """
        if self.transactions_df is None:
            raise ValueError("Must generate transactions first")
        
        # OPTIMIZED: Sort transactions by customer and date for efficient processing
        transactions_sorted = self.transactions_df.sort_values(
            by=['CustomerID', 'TransactionDate']
        ).copy()
        
        # OPTIMIZED: Pre-calculate cumulative spending per customer using groupby
        transactions_sorted['CumulativeSpend'] = transactions_sorted.groupby('CustomerID')['LineTotal'].cumsum()
        
        loyalty_records = []
        customer_points_balance = {}  # Track points balance per customer (OPTIMIZATION)
        record_id = 1
        
        # OPTIMIZED: Use itertuples() instead of iterrows() for better performance
        for row in transactions_sorted.itertuples():
            customer_id = row.CustomerID
            transaction_date = row.TransactionDate
            amount = row.LineTotal
            transaction_id = row.TransactionID
            
            # OPTIMIZED: Use pre-calculated cumulative spend instead of filtering
            total_spent = row.CumulativeSpend
            
            # Determine tier based on points (1 point per dollar, then tier multipliers)
            base_points = total_spent  # 1 point per dollar
            tier = self._get_tier_from_points(base_points)
            points_per_dollar = self._get_points_per_dollar(tier)
            
            # Points earned for this transaction
            points_earned = round(amount * points_per_dollar)
            
            # OPTIMIZED: Get previous points balance from dictionary (O(1))
            previous_points = customer_points_balance.get(customer_id, 0)
            
            # Random redemption (10% chance per transaction)
            points_redeemed = 0
            if np.random.random() < 0.1 and previous_points > 1000:
                points_redeemed = np.random.randint(100, min(previous_points, 5000))
            
            points_balance = max(0, previous_points + points_earned - points_redeemed)
            customer_points_balance[customer_id] = points_balance  # Update balance
            
            loyalty_records.append({
                'LoyaltyRecordID': record_id,
                'CustomerID': customer_id,
                'TransactionID': transaction_id,
                'TransactionDate': transaction_date,
                'PointsEarned': points_earned,
                'PointsRedeemed': points_redeemed,
                'PointsBalance': points_balance,
                'Tier': tier,
                'PointsPerDollar': points_per_dollar
            })
            
            record_id += 1
        
        self.loyalty_df = pd.DataFrame(loyalty_records)
        return self.loyalty_df
    
    def _get_tier_from_points(self, total_points: float) -> str:
        """Determine loyalty tier based on total points."""
        tiers = self.config['data_generation']['loyalty']['tiers']
        
        for tier in reversed(tiers):  # Check from highest to lowest
            if total_points >= tier['min_points']:
                return tier['name']
        return tiers[0]['name']  # Default to Bronze
    
    def _get_points_per_dollar(self, tier: str) -> float:
        """Get points per dollar for a given tier."""
        tiers = self.config['data_generation']['loyalty']['tiers']
        for t in tiers:
            if t['name'] == tier:
                return t['points_per_dollar']
        return 1.0
    
    def generate_survey_data(self) -> pd.DataFrame:
        """
        Generate customer survey data.
        
        Returns:
            DataFrame with survey responses
        """
        survey_frequency = self.config['data_generation']['survey']['survey_frequency']
        min_satisfaction = self.config['data_generation']['survey']['min_satisfaction_score']
        max_satisfaction = self.config['data_generation']['survey']['max_satisfaction_score']
        min_nps = self.config['data_generation']['survey']['min_nps_score']
        max_nps = self.config['data_generation']['survey']['max_nps_score']
        
        survey_records = []
        survey_id = 1
        
        # Select customers who have survey data
        num_surveys = int(self.num_customers * survey_frequency)
        surveyed_customers = np.random.choice(
            self.customers['CustomerID'].values,
            size=num_surveys,
            replace=False
        )
        
        for customer_id in surveyed_customers:
            # Get customer's transaction history to influence satisfaction
            customer_trans = self.transactions_df[
                self.transactions_df['CustomerID'] == customer_id
            ]
            
            if len(customer_trans) > 0:
                total_spent = customer_trans['LineTotal'].sum()
                num_transactions = len(customer_trans)
                
                # Higher spending customers tend to be more satisfied
                if total_spent > 5000:
                    satisfaction_bias = 0.3
                elif total_spent > 2000:
                    satisfaction_bias = 0.1
                else:
                    satisfaction_bias = -0.1
            else:
                satisfaction_bias = 0
            
            # Generate satisfaction score (1-5)
            base_satisfaction = np.random.uniform(min_satisfaction, max_satisfaction)
            satisfaction_score = max(
                min_satisfaction,
                min(max_satisfaction, round(base_satisfaction + satisfaction_bias))
            )
            
            # NPS score (0-10) correlated with satisfaction
            nps_base = (satisfaction_score - 1) / 4 * 10  # Map 1-5 to 0-10
            nps_score = max(
                min_nps,
                min(max_nps, round(nps_base + np.random.uniform(-2, 2)))
            )
            
            # Survey date (recent, after some transactions)
            if len(customer_trans) > 0:
                last_trans_date = pd.to_datetime(customer_trans['TransactionDate'].max())
                survey_date = last_trans_date + timedelta(days=np.random.randint(1, 90))
            else:
                survey_date = self.fake.date_between(
                    start_date=self.start_date.date(),
                    end_date=self.end_date.date()
                )
            
            # Generate feedback text
            feedback_options = [
                "Great experience, will shop again!",
                "Good service and product quality.",
                "Satisfied with my purchase.",
                "Could be better, but acceptable.",
                "Not very satisfied, needs improvement.",
                "Excellent customer service!",
                "Products are good value for money.",
                "Average experience overall.",
                "Love the loyalty program benefits!",
                "Would recommend to friends."
            ]
            
            survey_records.append({
                'SurveyID': survey_id,
                'CustomerID': customer_id,
                'SurveyDate': survey_date,
                'SatisfactionScore': satisfaction_score,
                'NPSScore': nps_score,
                'Feedback': np.random.choice(feedback_options),
                'WouldRecommend': 'Yes' if nps_score >= 7 else 'No'
            })
            
            survey_id += 1
        
        self.survey_df = pd.DataFrame(survey_records)
        return self.survey_df
    
    def generate_all_datasets(self, output_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
        """
        Generate all datasets and save to files.
        
        Args:
            output_dir: Directory to save generated data files
            
        Returns:
            Dictionary with all generated DataFrames
        """
        print("Generating products...")
        self.generate_products()
        
        print("Generating stores...")
        self.generate_stores()
        
        print("Generating customers...")
        self.generate_customers()
        
        print("Generating transactions...")
        self.generate_transactions()
        
        print("Generating loyalty data...")
        self.generate_loyalty_data()
        
        print("Generating survey data...")
        self.generate_survey_data()
        
        # Save to files
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving datasets to {output_dir}...")
        self.transactions_df.to_csv(output_path / 'transactions.csv', index=False)
        self.loyalty_df.to_csv(output_path / 'loyalty_pc_optimum.csv', index=False)
        self.survey_df.to_csv(output_path / 'surveys.csv', index=False)
        self.products.to_csv(output_path / 'products.csv', index=False)
        self.stores.to_csv(output_path / 'stores.csv', index=False)
        self.customers.to_csv(output_path / 'customers.csv', index=False)
        
        print("Data generation complete!")
        print(f"\nGenerated datasets:")
        print(f"  - Transactions: {len(self.transactions_df):,} records")
        print(f"  - Loyalty: {len(self.loyalty_df):,} records")
        print(f"  - Surveys: {len(self.survey_df):,} records")
        print(f"  - Products: {len(self.products):,} products")
        print(f"  - Stores: {len(self.stores):,} stores")
        print(f"  - Customers: {len(self.customers):,} customers")
        
        return {
            'transactions': self.transactions_df,
            'loyalty': self.loyalty_df,
            'surveys': self.survey_df,
            'products': self.products,
            'stores': self.stores,
            'customers': self.customers
        }


def generate_all_datasets(config_path: str = "config/data_generation_config.yaml",
                         output_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to generate all datasets.
    
    Args:
        config_path: Path to configuration YAML file
        output_dir: Directory to save generated data files
        
    Returns:
        Dictionary with all generated DataFrames
    """
    generator = SyntheticDataGenerator(config_path)
    return generator.generate_all_datasets(output_dir)


if __name__ == "__main__":
    # Generate all datasets
    datasets = generate_all_datasets()
    print("\nData generation completed successfully!")
