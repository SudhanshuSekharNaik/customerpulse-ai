"""Generate the standardized Amazon E-Commerce Benchmark Dataset.
Contains 12,000 order transactions across 1,500 unique customer accounts.
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_amazon_demo_csv():
    os.makedirs("data", exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    n_orders = 12000
    n_customers = 1500

    customer_ids = [f"CUST_{i+1:05d}" for i in range(n_customers)]
    cities = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad", "Jaipur"]
    cust_city_map = {cid: random.choice(cities) for cid in customer_ids}

    # Categories and items
    catalog = {
        "Smartphones & Electronics": [
            ("PROD_E101", "Apple iPhone 15 Pro", 129990.0),
            ("PROD_E102", "Samsung Galaxy S24 Ultra", 119999.0),
            ("PROD_E103", "OnePlus 12 5G", 64999.0),
            ("PROD_E104", "Sony WH-1000XM5 ANC Headphones", 29990.0),
            ("PROD_E105", "Apple iPad Air M2", 59900.0),
            ("PROD_E106", "Anker 65W Fast Charger", 2499.0),
        ],
        "Fashion & Apparel": [
            ("PROD_F201", "Levi's Men 511 Slim Fit Jeans", 3499.0),
            ("PROD_F202", "Biba Women Anarkali Kurta Set", 4999.0),
            ("PROD_F203", "Nike Air Max Running Shoes", 8995.0),
            ("PROD_F204", "Puma Casual Cotton Hoodie", 2799.0),
            ("PROD_F205", "Ray-Ban Aviator Sunglasses", 7590.0),
        ],
        "Home Appliances": [
            ("PROD_H301", "Dyson V12 Cordless Vacuum Cleaner", 47900.0),
            ("PROD_H302", "Philips Digital Air Fryer 4.1L", 7495.0),
            ("PROD_H303", "LG 28L Convection Microwave Oven", 14490.0),
            ("PROD_H304", "Eureka Forbes Water Purifier", 12999.0),
        ],
        "Beauty & Personal Care": [
            ("PROD_B401", "Forest Essentials Facial Toner", 1450.0),
            ("PROD_B402", "Minimalist 10% Niacinamide Serum", 599.0),
            ("PROD_B403", "L'Oreal Paris Hair Masque", 899.0),
            ("PROD_B404", "Philips Multigroomer Trimmer", 2195.0),
        ],
        "Books & Stationery": [
            ("PROD_K501", "Atomic Habits by James Clear", 499.0),
            ("PROD_K502", "Psychology of Money", 399.0),
            ("PROD_K503", "Moleskine Hardcover Journal", 1890.0),
        ],
    }

    categories = list(catalog.keys())
    payment_methods = ["UPI / GPay / PhonePe", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"]
    channels = ["Amazon Mobile App", "Amazon Web Browser", "Affiliate / Partner Link", "Push Notification Deal"]

    # Generate dates spanning past 180 days (Oct 2025 - Mar 2026)
    end_date = datetime(2026, 3, 15, 22, 0, 0)
    start_date = end_date - timedelta(days=180)

    rows = []
    # Assign higher order frequencies to top customers to form realistic power-law / RFM distribution
    cust_weights = np.random.pareto(a=1.8, size=n_customers) + 0.2
    cust_weights /= cust_weights.sum()

    assigned_customers = np.random.choice(customer_ids, size=n_orders, p=cust_weights)

    for i in range(n_orders):
        cid = assigned_customers[i]
        city = cust_city_map[cid]

        # Random timestamp with peak shopping weights (7 PM - 11 PM and 1 PM - 3 PM)
        rand_day_offset = random.uniform(0, 180)
        base_dt = start_date + timedelta(days=rand_day_offset)
        hour_p = [0.01, 0.005, 0.002, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.07,
                  0.08, 0.09, 0.08, 0.06, 0.05, 0.06, 0.08, 0.10, 0.11, 0.09, 0.06, 0.03]
        hour_p = [p / sum(hour_p) for p in hour_p]
        chosen_hour = np.random.choice(range(24), p=hour_p)
        chosen_minute = random.randint(0, 59)
        chosen_second = random.randint(0, 59)
        order_date = base_dt.replace(hour=chosen_hour, minute=chosen_minute, second=chosen_second)

        # Product selection
        cat_p = [0.35, 0.30, 0.15, 0.12, 0.08]
        chosen_cat = np.random.choice(categories, p=cat_p)
        prod_id, prod_name, unit_price = random.choice(catalog[chosen_cat])

        quantity = 1 if unit_price > 10000 else np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05])
        discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05])
        raw_amount = unit_price * quantity
        discount_amount = round(raw_amount * (discount_pct / 100.0), 2)
        sales_amount = round(raw_amount - discount_amount, 2)

        delivery_days = random.randint(1, 5)
        rating = np.random.choice([5, 4, 3, 2, 1], p=[0.60, 0.25, 0.08, 0.04, 0.03])
        returned = 1 if (rating <= 2 and random.random() < 0.6) or random.random() < 0.03 else 0

        rows.append({
            "order_id": f"ORD_{i+100001:07d}",
            "customer_id": cid,
            "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
            "product_id": prod_id,
            "product_category": chosen_cat,
            "quantity": int(quantity),
            "unit_price": float(unit_price),
            "discount_percent": int(discount_pct),
            "discount_amount": float(discount_amount),
            "sales_amount": float(sales_amount),
            "payment_method": random.choice(payment_methods),
            "channel": random.choice(channels),
            "customer_city": city,
            "delivery_days": int(delivery_days),
            "rating": int(rating),
            "returned": int(returned),
        })

    df = pd.DataFrame(rows)
    # Sort chronologically
    df = df.sort_values(by="order_date").reset_index(drop=True)
    out_path = "data/amazon_ecommerce_demo.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} orders across {df['customer_id'].nunique()} unique customers -> {out_path}")
    print(f"Total Sales Revenue: INR {df['sales_amount'].sum():,.2f}")
    return out_path

if __name__ == "__main__":
    generate_amazon_demo_csv()
