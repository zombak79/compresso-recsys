"""Measured Amazon category/protocol defaults; unlisted cases use the baseline.

Profiles are individually verified on metadata-eligible all-rating feedback,
seed 42, one evaluation draw and 339-day temporal windows. Source fingerprints,
measured counts and small-evaluation warnings are recorded in
docs/source/_static/amazon-default-profiles.json. All 33 named categories have
four verified profiles. Clothing temporal uses the reviewed 120k-user/25k-item
size exception; all other profiles retain their original category limits.
"""

AMAZON_SPLIT_DEFAULTS = {
    "All_Beauty": {
        "user_split": dict(min_user_support=2, item_min_support=2, val_users=2263, test_users=2263),
        "item_split": dict(min_user_support=2, item_min_support=2),
        "leave_last_out": dict(min_user_support=4, item_min_support=1),
        "temporal": dict(min_user_support=3, item_min_support=1),
    },
    "Amazon_Fashion": {
        "user_split": dict(min_user_support=2, item_min_support=4, val_users=3571, test_users=3571),
        "item_split": dict(min_user_support=2, item_min_support=4),
        "leave_last_out": dict(min_user_support=4, item_min_support=2),
        "temporal": dict(min_user_support=2, item_min_support=2),
    },
    "Appliances": {
        "user_split": dict(min_user_support=3, item_min_support=2, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=3, item_min_support=2),
        "leave_last_out": dict(min_user_support=4, item_min_support=2),
        "temporal": dict(min_user_support=2, item_min_support=2),
    },
    "Arts_Crafts_and_Sewing": {
        "user_split": dict(min_user_support=5, item_min_support=16, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=5, item_min_support=16),
        "leave_last_out": dict(min_user_support=5, item_min_support=16),
        "temporal": dict(min_user_support=3, item_min_support=12),
    },
    "Automotive": {
        "user_split": dict(min_user_support=7, item_min_support=25, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=7, item_min_support=25),
        "leave_last_out": dict(min_user_support=7, item_min_support=25),
        "temporal": dict(min_user_support=6, item_min_support=12),
    },
    "Baby_Products": {
        "user_split": dict(min_user_support=6, item_min_support=9, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=6, item_min_support=9),
        "leave_last_out": dict(min_user_support=6, item_min_support=9),
        "temporal": dict(min_user_support=3, item_min_support=7),
    },
    "Beauty_and_Personal_Care": {
        "user_split": dict(min_user_support=8, item_min_support=25, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=8, item_min_support=25),
        "leave_last_out": dict(min_user_support=8, item_min_support=25),
        "temporal": dict(min_user_support=6, item_min_support=18),
    },
    "Books": {
        "user_split": dict(min_user_support=5, item_min_support=17, val_users=20000, test_users=20000),
        "item_split": dict(min_user_support=5, item_min_support=17),
        "leave_last_out": dict(min_user_support=5, item_min_support=17),
        "temporal": dict(min_user_support=6, item_min_support=6),
    },
    "CDs_and_Vinyl": {
        "user_split": dict(min_user_support=5, item_min_support=16, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=5, item_min_support=16),
        "leave_last_out": dict(min_user_support=5, item_min_support=16),
        "temporal": dict(min_user_support=2, item_min_support=5),
    },
    "Cell_Phones_and_Accessories": {
        "user_split": dict(min_user_support=6, item_min_support=13, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=6, item_min_support=13),
        "leave_last_out": dict(min_user_support=6, item_min_support=13),
        "temporal": dict(min_user_support=5, item_min_support=8),
    },
    "Clothing_Shoes_and_Jewelry": {
        "user_split": dict(min_user_support=10, item_min_support=35, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=10, item_min_support=35),
        "leave_last_out": dict(min_user_support=10, item_min_support=35),
        "temporal": dict(min_user_support=10, item_min_support=17),
    },
    "Digital_Music": {
        "user_split": dict(min_user_support=2, item_min_support=2, val_users=278, test_users=278),
        "item_split": dict(min_user_support=2, item_min_support=2),
        "leave_last_out": dict(min_user_support=4, item_min_support=1),
        "temporal": dict(min_user_support=2, item_min_support=1),
    },
    "Electronics": {
        "user_split": dict(min_user_support=8, item_min_support=16, val_users=20000, test_users=20000),
        "item_split": dict(min_user_support=8, item_min_support=16),
        "leave_last_out": dict(min_user_support=8, item_min_support=16),
        "temporal": dict(min_user_support=6, item_min_support=12),
    },
    "Gift_Cards": {
        "user_split": dict(min_user_support=2, item_min_support=1, val_users=1142, test_users=1142),
        "item_split": dict(min_user_support=2, item_min_support=2),
        "leave_last_out": dict(min_user_support=4, item_min_support=1),
        "temporal": dict(min_user_support=2, item_min_support=1),
    },
    "Grocery_and_Gourmet_Food": {
        "user_split": dict(min_user_support=7, item_min_support=24, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=7, item_min_support=24),
        "leave_last_out": dict(min_user_support=7, item_min_support=24),
        "temporal": dict(min_user_support=6, item_min_support=13),
    },
    "Handmade_Products": {
        "user_split": dict(min_user_support=2, item_min_support=2, val_users=2215, test_users=2215),
        "item_split": dict(min_user_support=2, item_min_support=2),
        "leave_last_out": dict(min_user_support=4, item_min_support=1),
        "temporal": dict(min_user_support=3, item_min_support=1),
    },
    "Health_and_Household": {
        "user_split": dict(min_user_support=10, item_min_support=20, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=10, item_min_support=20),
        "leave_last_out": dict(min_user_support=10, item_min_support=20),
        "temporal": dict(min_user_support=8, item_min_support=15),
    },
    "Health_and_Personal_Care": {
        "user_split": dict(min_user_support=2, item_min_support=1, val_users=1839, test_users=1839),
        "item_split": dict(min_user_support=2, item_min_support=1),
        "leave_last_out": dict(min_user_support=4, item_min_support=1),
        "temporal": dict(min_user_support=2, item_min_support=1),
    },
    "Home_and_Kitchen": {
        "user_split": dict(min_user_support=12, item_min_support=35, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=12, item_min_support=35),
        "leave_last_out": dict(min_user_support=12, item_min_support=35),
        "temporal": dict(min_user_support=10, item_min_support=25),
    },
    "Industrial_and_Scientific": {
        "user_split": dict(min_user_support=4, item_min_support=8, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=4, item_min_support=8),
        "leave_last_out": dict(min_user_support=4, item_min_support=8),
        "temporal": dict(min_user_support=3, item_min_support=5),
    },
    "Kindle_Store": {
        "user_split": dict(min_user_support=12, item_min_support=76, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=12, item_min_support=76),
        "leave_last_out": dict(min_user_support=12, item_min_support=76),
        "temporal": dict(min_user_support=8, item_min_support=50),
    },
    "Magazine_Subscriptions": {
        "user_split": dict(min_user_support=2, item_min_support=1, val_users=1200, test_users=1200),
        "item_split": dict(min_user_support=2, item_min_support=2),
        "leave_last_out": dict(min_user_support=4, item_min_support=1),
        "temporal": dict(min_user_support=2, item_min_support=1),
    },
    "Movies_and_TV": {
        "user_split": dict(min_user_support=10, item_min_support=36, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=10, item_min_support=36),
        "leave_last_out": dict(min_user_support=10, item_min_support=36),
        "temporal": dict(min_user_support=5, item_min_support=12),
    },
    "Musical_Instruments": {
        "user_split": dict(min_user_support=5, item_min_support=7, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=5, item_min_support=7),
        "leave_last_out": dict(min_user_support=5, item_min_support=7),
        "temporal": dict(min_user_support=2, item_min_support=5),
    },
    "Office_Products": {
        "user_split": dict(min_user_support=5, item_min_support=18, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=5, item_min_support=18),
        "leave_last_out": dict(min_user_support=5, item_min_support=18),
        "temporal": dict(min_user_support=4, item_min_support=8),
    },
    "Patio_Lawn_and_Garden": {
        "user_split": dict(min_user_support=6, item_min_support=21, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=6, item_min_support=21),
        "leave_last_out": dict(min_user_support=6, item_min_support=21),
        "temporal": dict(min_user_support=5, item_min_support=13),
    },
    "Pet_Supplies": {
        "user_split": dict(min_user_support=10, item_min_support=16, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=10, item_min_support=16),
        "leave_last_out": dict(min_user_support=10, item_min_support=16),
        "temporal": dict(min_user_support=8, item_min_support=12),
    },
    "Software": {
        "user_split": dict(min_user_support=6, item_min_support=6, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=6, item_min_support=6),
        "leave_last_out": dict(min_user_support=6, item_min_support=6),
        "temporal": dict(min_user_support=2, item_min_support=2),
    },
    "Sports_and_Outdoors": {
        "user_split": dict(min_user_support=6, item_min_support=17, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=6, item_min_support=17),
        "leave_last_out": dict(min_user_support=6, item_min_support=17),
        "temporal": dict(min_user_support=5, item_min_support=9),
    },
    "Subscription_Boxes": {
        "user_split": dict(min_user_support=2, item_min_support=1, val_users=57, test_users=57),
        "item_split": dict(min_user_support=2, item_min_support=1),
        "leave_last_out": dict(min_user_support=4, item_min_support=1),
        "temporal": dict(min_user_support=2, item_min_support=1),
    },
    "Tools_and_Home_Improvement": {
        "user_split": dict(min_user_support=8, item_min_support=26, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=8, item_min_support=26),
        "leave_last_out": dict(min_user_support=8, item_min_support=26),
        "temporal": dict(min_user_support=6, item_min_support=18),
    },
    "Toys_and_Games": {
        "user_split": dict(min_user_support=6, item_min_support=22, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=6, item_min_support=22),
        "leave_last_out": dict(min_user_support=6, item_min_support=22),
        "temporal": dict(min_user_support=5, item_min_support=10),
    },
    "Video_Games": {
        "user_split": dict(min_user_support=5, item_min_support=7, val_users=5000, test_users=5000),
        "item_split": dict(min_user_support=5, item_min_support=7),
        "leave_last_out": dict(min_user_support=5, item_min_support=7),
        "temporal": dict(min_user_support=3, item_min_support=4),
    },
}
