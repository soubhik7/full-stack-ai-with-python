order_amount = int(input("Enter the order amount: "))
#order_amount2 = input("Enter the order amount:  ")

#print(f"order_amount is of type: {type(order_amount2)}")


delivery_fees = 0 if order_amount > 300 else 30 # Conditional (ternary) operator

print(f"Delivery fees is : {delivery_fees}")