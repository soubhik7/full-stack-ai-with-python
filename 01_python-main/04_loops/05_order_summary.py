names = ["Hitesh", "Meera", "Sam", "Ali"]
bills = [50, 70, 100, 55]

for name, amount in zip(names, bills):     # Using zip to iterate over both lists simultaneously
    print(f"{name} paid {amount} rupees")
    