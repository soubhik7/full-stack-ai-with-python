users = [
    {"id": 1, "total": 100, "coupon": "P20"},
    {"id": 2, "total": 150, "coupon": "F10"},
    {"id": 3, "total": 80, "coupon": "P50"},
    {"id": 4, "total": 200, "coupon": "NONE"},
    {"id": 5, "total": 50, "coupon": "P20"},
]

discounts = {
    "P20": (0.2, 0),
    "F10": (0.5, 0),
    "P50": (0, 10),
    "NONE": (0, 0),
    "NEWUSER": (0.3, 5),
    "VIP": (0.4, 20),
}

for user in users:
    percent, fixed = discounts.get(user["coupon"], (0, 0))
    discount = user["total"] * percent + fixed
    print(f"{user["id"]} paid {user["total"]} and got discount for next visit of rupees {discount}")