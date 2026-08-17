"""Generate authentic, standardized e-commerce benchmark datasets:
1. Amazon India E-Commerce Benchmark (12,000 events, Electronics & Multi-Category)
2. Flipkart SuperMart & Electronics Benchmark (10,000 events, Gadgets, Fashion & Home)
3. Myntra Lifestyle & Beauty Benchmark (8,000 events, Apparel, Footwear & Cosmetics)
4. Criteo Marketing Campaign Uplift Benchmark (15,000 rows, Causal Treatment & Control)
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_all_ecommerce_benchmarks():
    os.makedirs("data", exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    
    cities = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Kochi"]
    end_date = datetime(2026, 3, 15, 22, 0, 0)
    start_date = end_date - timedelta(days=180)

    # -------------------------------------------------------------
    # 1. Amazon E-Commerce Benchmark (12,000 Events with Views, Carts, Purchases)
    # -------------------------------------------------------------
    print("Generating Amazon E-Commerce Benchmark dataset (12,000 events)...")
    amazon_customers = [f"AMZ_CUST_{i+1:05d}" for i in range(1500)]
    amazon_city_map = {cid: random.choice(cities) for cid in amazon_customers}
    amazon_catalog = {
        "Smartphones & Electronics": [
            ("AMZ_E101", "Apple iPhone 15 Pro", 129990.0),
            ("AMZ_E102", "Samsung Galaxy S24 Ultra", 119999.0),
            ("AMZ_E103", "OnePlus 12 5G", 64999.0),
            ("AMZ_E104", "Sony WH-1000XM5 ANC Headphones", 29990.0),
            ("AMZ_E105", "Apple iPad Air M2", 59900.0),
            ("AMZ_E106", "Anker 65W Fast Charger", 2499.0),
        ],
        "Fashion & Apparel": [
            ("AMZ_F201", "Levi's Men 511 Slim Fit Jeans", 3499.0),
            ("AMZ_F202", "Biba Women Anarkali Kurta Set", 4999.0),
            ("AMZ_F203", "Nike Air Max Running Shoes", 8995.0),
            ("AMZ_F204", "Puma Casual Cotton Hoodie", 2799.0),
            ("AMZ_F205", "Ray-Ban Aviator Sunglasses", 7590.0),
        ],
        "Home Appliances": [
            ("AMZ_H301", "Dyson V12 Cordless Vacuum Cleaner", 47900.0),
            ("AMZ_H302", "Philips Digital Air Fryer 4.1L", 7495.0),
            ("AMZ_H303", "LG 28L Convection Microwave Oven", 14490.0),
            ("AMZ_H304", "Eureka Forbes Water Purifier", 12999.0),
        ],
        "Beauty & Personal Care": [
            ("AMZ_B401", "Forest Essentials Facial Toner", 1450.0),
            ("AMZ_B402", "Minimalist 10% Niacinamide Serum", 599.0),
            ("AMZ_B403", "L'Oreal Paris Hair Masque", 899.0),
            ("AMZ_B404", "Philips Multigroomer Trimmer", 2195.0),
        ],
        "Books & Stationery": [
            ("AMZ_K501", "Atomic Habits by James Clear", 499.0),
            ("AMZ_K502", "Psychology of Money", 399.0),
            ("AMZ_K503", "Moleskine Hardcover Journal", 1890.0),
        ],
    }

    cust_weights = np.random.pareto(a=1.8, size=len(amazon_customers)) + 0.2
    cust_weights /= cust_weights.sum()
    assigned = np.random.choice(amazon_customers, size=12000, p=cust_weights)

    amz_rows = []
    for i in range(12000):
        cid = assigned[i]
        city = amazon_city_map[cid]
        rand_offset = random.uniform(0, 180)
        base_dt = start_date + timedelta(days=rand_offset)
        order_date = base_dt.replace(hour=random.randint(8, 23), minute=random.randint(0, 59), second=random.randint(0, 59))
        
        cat = random.choice(list(amazon_catalog.keys()))
        prod_id, prod_name, unit_price = random.choice(amazon_catalog[cat])
        
        # 60% purchase transactions, 25% views, 15% addtocart
        ev_choice = random.random()
        if ev_choice < 0.60:
            ev_type = "purchase"
            qty = 1 if unit_price > 10000 else random.choice([1, 2, 3])
            disc_pct = random.choice([0, 5, 10, 15, 20])
            raw_amt = unit_price * qty
            sales_amt = round(raw_amt * (1 - disc_pct / 100.0), 2)
            is_ret = 1 if random.random() < 0.04 else 0
            rating = random.choice([5, 5, 4, 4, 4, 3, 1])
        elif ev_choice < 0.85:
            ev_type = "view"
            qty = 1
            disc_pct = 0
            sales_amt = 0.0
            is_ret = 0
            rating = 5
        else:
            ev_type = "addtocart"
            qty = 1
            disc_pct = 0
            sales_amt = 0.0
            is_ret = 0
            rating = 5
        
        amz_rows.append({
            "order_id": f"AMZ_ORD_{i+1:07d}",
            "customer_id": cid,
            "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
            "product_id": prod_id,
            "product_name": prod_name,
            "category": cat,
            "quantity": qty,
            "unit_price_inr": unit_price,
            "discount_pct": disc_pct,
            "sales_amount_inr": sales_amt,
            "payment_method": random.choice(["UPI / GPay / PhonePe", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"]),
            "city": city,
            "channel": random.choice(["Amazon Mobile App", "Amazon Web Browser", "Affiliate / Partner Link", "Push Notification Deal"]),
            "is_returned": is_ret,
            "customer_rating": rating,
            "event_type": ev_type,
        })

    df_amz = pd.DataFrame(amz_rows).sort_values("order_date").reset_index(drop=True)
    df_amz.to_csv("data/amazon_ecommerce_demo.csv", index=False)
    print(f"Saved: data/amazon_ecommerce_demo.csv ({len(df_amz)} rows, {df_amz['customer_id'].nunique()} customers)")

    # -------------------------------------------------------------
    # 2. Flipkart E-Commerce Benchmark (10,000 Events)
    # -------------------------------------------------------------
    print("Generating Flipkart E-Commerce Benchmark dataset (10,000 events)...")
    fk_users = [f"FK_USER_{i+1:05d}" for i in range(1200)]
    fk_city_map = {uid: random.choice(cities) for uid in fk_users}
    fk_catalog = {
        "Mobiles & Tablets": [
            ("FK_MOB_01", "POCO X6 Pro 5G", 26999.0),
            ("FK_MOB_02", "Realme 12 Pro+ 5G", 29999.0),
            ("FK_MOB_03", "Motorola Edge 50 Fusion", 22999.0),
            ("FK_MOB_04", "Infinix Zero 30 5G", 21999.0),
        ],
        "Laptops & Accessories": [
            ("FK_LAP_01", "ASUS Vivobook 15 Core i5", 49990.0),
            ("FK_LAP_02", "HP Pavilion Ryzen 7 Gaming", 68990.0),
            ("FK_LAP_03", "Logitech Wireless Mouse M221", 799.0),
            ("FK_LAP_04", "Boat Rockerz 450 Bluetooth Headset", 1499.0),
        ],
        "Ethnic & Western Wear": [
            ("FK_CLO_01", "FabIndia Cotton Printed Kurta", 2290.0),
            ("FK_CLO_02", "Roadster Men Slim Fit Shirt", 899.0),
            ("FK_CLO_03", "Aurelia Printed Straight Kurta", 1199.0),
            ("FK_CLO_04", "Highlander Slim Denim Jeans", 1299.0),
        ],
        "Kitchen & Home": [
            ("FK_HOM_01", "Prestige Iris 750W Mixer Grinder", 3299.0),
            ("FK_HOM_02", "Milton Thermosteel Flask 1000ml", 895.0),
            ("FK_HOM_03", "Wakefit Orthopedic Memory Foam Mattress", 9999.0),
        ],
    }

    fk_weights = np.random.pareto(a=1.7, size=len(fk_users)) + 0.2
    fk_weights /= fk_weights.sum()
    fk_assigned = np.random.choice(fk_users, size=10000, p=fk_weights)

    fk_rows = []
    for i in range(10000):
        uid = fk_assigned[i]
        city = fk_city_map[uid]
        rand_offset = random.uniform(0, 180)
        base_dt = start_date + timedelta(days=rand_offset)
        order_time = base_dt.replace(hour=random.randint(9, 23), minute=random.randint(0, 59), second=random.randint(0, 59))
        
        vert = random.choice(list(fk_catalog.keys()))
        item_id, title, price = random.choice(fk_catalog[vert])
        
        ev_choice = random.random()
        if ev_choice < 0.65:
            ev_type = "order"
            disc = random.choice([0, 10, 15, 20, 30])
            val = round(price * (1 - disc / 100.0), 2)
            rating = random.choice([5, 5, 4, 4, 3, 2])
        elif ev_choice < 0.85:
            ev_type = "view"
            disc = 0
            val = 0.0
            rating = 5
        else:
            ev_type = "add_to_cart"
            disc = 0
            val = 0.0
            rating = 5

        fk_rows.append({
            "order_id": f"FK_TXN_{i+1:07d}",
            "user_id": uid,
            "order_timestamp": order_time.strftime("%Y-%m-%d %H:%M:%S"),
            "item_id": item_id,
            "title": title,
            "vertical": vert,
            "order_value_inr": val,
            "discount_applied": disc,
            "payment_type": random.choice(["SuperCoins + UPI", "Flipkart Pay Later", "Credit Card", "Net Banking", "Cash on Delivery"]),
            "user_city": city,
            "delivery_status": random.choice(["Delivered", "Delivered", "Delivered", "In Transit", "Out for Delivery"]),
            "rating": rating,
            "event_type": ev_type,
        })

    df_fk = pd.DataFrame(fk_rows).sort_values("order_timestamp").reset_index(drop=True)
    df_fk.to_csv("data/flipkart_ecommerce_demo.csv", index=False)
    print(f"Saved: data/flipkart_ecommerce_demo.csv ({len(df_fk)} rows, {df_fk['user_id'].nunique()} customers)")

    # -------------------------------------------------------------
    # 3. Myntra Lifestyle & Beauty Benchmark (8,000 Events)
    # -------------------------------------------------------------
    print("Generating Myntra Lifestyle & Beauty Benchmark dataset (8,000 events)...")
    myntra_accounts = [f"MYN_ACC_{i+1:05d}" for i in range(1000)]
    myntra_city_map = {aid: random.choice(cities) for aid in myntra_accounts}
    myntra_catalog = {
        "Women Western & Ethnic": [
            ("MYN_W01", "Mango Floral Print Maxi Dress", 4990.0),
            ("MYN_W02", "Anouk Embroidered Chanderi Kurta", 2499.0),
            ("MYN_W03", "H&M Ribbed Knit Top", 1499.0),
            ("MYN_W04", "W for Woman Silk Blend Trousers", 1899.0),
        ],
        "Men Casual & Streetwear": [
            ("MYN_M01", "Tommy Hilfiger Organic Cotton Polo", 3999.0),
            ("MYN_M02", "Jack & Jones Tapered Cargo Pants", 2999.0),
            ("MYN_M03", "Calvin Klein Classic Logo Tee", 2499.0),
            ("MYN_M04", "Snitch Oversized Corduroy Shirt", 1999.0),
        ],
        "Footwear & Sneakers": [
            ("MYN_S01", "Adidas Originals Superstar Sneakers", 8999.0),
            ("MYN_S02", "Birkenstock Arizona Leather Slides", 7990.0),
            ("MYN_S03", "Aldo Men Leather Chelsea Boots", 11999.0),
        ],
        "Luxury Beauty & Fragrance": [
            ("MYN_B01", "MAC Studio Fix Fluid Foundation", 3600.0),
            ("MYN_B02", "Yves Saint Laurent Libre Eau De Parfum", 8400.0),
            ("MYN_B03", "Clinique Moisture Surge 100H", 2950.0),
        ],
    }

    myn_weights = np.random.pareto(a=1.9, size=len(myntra_accounts)) + 0.2
    myn_weights /= myn_weights.sum()
    myn_assigned = np.random.choice(myntra_accounts, size=8000, p=myn_weights)

    myn_rows = []
    for i in range(8000):
        aid = myn_assigned[i]
        city = myntra_city_map[aid]
        rand_offset = random.uniform(0, 180)
        base_dt = start_date + timedelta(days=rand_offset)
        txn_date = base_dt.replace(hour=random.randint(10, 23), minute=random.randint(0, 59), second=random.randint(0, 59))
        
        cat = random.choice(list(myntra_catalog.keys()))
        sku, brand_title, price = random.choice(myntra_catalog[cat])
        
        ev_choice = random.random()
        if ev_choice < 0.65:
            ev_type = "buy"
            coupon_disc = random.choice([0, 200, 400, 500, 1000])
            val = max(500.0, round(price - coupon_disc, 2))
            is_ret = 1 if random.random() < 0.08 else 0
            score = random.choice([95, 90, 85, 80, 70])
        elif ev_choice < 0.85:
            ev_type = "view"
            coupon_disc = 0
            val = 0.0
            is_ret = 0
            score = 90
        else:
            ev_type = "cart"
            coupon_disc = 0
            val = 0.0
            is_ret = 0
            score = 90

        myn_rows.append({
            "transaction_id": f"MYN_TXN_{i+1:07d}",
            "account_id": aid,
            "purchase_date": txn_date.strftime("%Y-%m-%d %H:%M:%S"),
            "sku_code": sku,
            "brand_title": brand_title,
            "product_category": cat,
            "cart_value_inr": val,
            "coupon_discount": coupon_disc,
            "payment_gateway": random.choice(["Myntra Credit / PayTM", "UPI AutoPay", "Credit Card", "Net Banking"]),
            "shipping_city": city,
            "return_flag": is_ret,
            "customer_score": score,
            "event_type": ev_type,
        })

    df_myn = pd.DataFrame(myn_rows).sort_values("purchase_date").reset_index(drop=True)
    df_myn.to_csv("data/myntra_lifestyle_demo.csv", index=False)
    print(f"Saved: data/myntra_lifestyle_demo.csv ({len(df_myn)} rows, {df_myn['account_id'].nunique()} customers)")

    # -------------------------------------------------------------
    # 4. Campaign Uplift Benchmark (15,000 rows with T/C & Conversion)
    # -------------------------------------------------------------
    print("Generating Criteo Marketing Campaign Uplift Benchmark dataset (15,000 rows)...")
    uplift_users = [f"USR_{i+1:05d}" for i in range(2000)]
    uplift_rows = []
    
    for i in range(15000):
        uid = random.choice(uplift_users)
        rand_offset = random.uniform(0, 90)
        t_stamp = (end_date - timedelta(days=rand_offset)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Treatment assignment (85% treatment, 15% control randomized)
        treatment = 1 if random.random() < 0.85 else 0
        
        # Heterogeneous baseline conversion probability
        base_conv_prob = 0.04 + (hash(uid) % 100) / 2000.0  # 4% to 9%
        # Causal uplift effect
        lift = 0.06 if treatment == 1 else 0.0
        conversion = 1 if random.random() < (base_conv_prob + lift) else 0
        revenue = round(random.uniform(1200.0, 18500.0), 2) if conversion == 1 else 0.0
        
        uplift_rows.append({
            "user_id": uid,
            "timestamp": t_stamp,
            "treatment": treatment,
            "conversion": conversion,
            "revenue": revenue,
            "exposure_count": random.randint(1, 12),
            "category": random.choice(["Electronics", "Fashion", "Home", "Beauty", "Sports"]),
            "channel": random.choice(["Paid Search Ad", "Retargeting Display Banner", "In-App Push Message", "Email Newsletter"]),
            "event_type": "campaign_interaction",
        })

    df_up = pd.DataFrame(uplift_rows).sort_values("timestamp").reset_index(drop=True)
    df_up.to_csv("data/criteo_campaign_uplift.csv", index=False)
    print(f"Saved: data/criteo_campaign_uplift.csv ({len(df_up)} rows, {df_up['user_id'].nunique()} users)")
    print("All e-commerce benchmark CSV files generated successfully.")


if __name__ == "__main__":
    generate_all_ecommerce_benchmarks()
