# Module 04 — Loops

**Difficulty:** Beginner | **Time:** 2–3 hours | **Prerequisites:** Module 03

---

## Why This Module Matters

Loops eliminate repetition. Instead of writing `print("Serving token 1")` ten times, you write it once inside a loop. Any task that involves processing a collection, repeating an action, or waiting for a condition belongs in a loop.

Loops are how programs become powerful — a program that processes 10 items works the same way for 10,000 items.

---

## Types of Loops in Python

| Loop Type | Syntax | Use When |
|-----------|--------|---------|
| `for` | `for item in collection:` | You know the collection or range upfront |
| `while` | `while condition:` | You loop until some condition becomes `False` |

---

## Core Concepts

| Concept | What It Does |
|---------|-------------|
| `range(n)` | Generates numbers 0 to n-1 |
| `range(start, stop)` | Generates numbers from start to stop-1 |
| `range(start, stop, step)` | Generates numbers with a step increment |
| `break` | Exits the loop immediately |
| `continue` | Skips the rest of the current iteration, goes to next |
| `for/else` | `else` block runs if the loop completes without `break` |
| Walrus `:=` | Assigns a value AND uses it in the same expression |

---

## `for` Loop — How It Works

```python
# Iterating over a range
for token in range(1, 6):
    print(f"Serving token #{token}")
# Output: 1, 2, 3, 4, 5

# Iterating over a list
flavors = ["Masala", "Ginger", "Elaichi"]
for flavor in flavors:
    print(f"Today's special: {flavor}")
```

---

## `while` Loop — How It Works

```python
cups_remaining = 3
while cups_remaining > 0:
    print(f"Serving cup, {cups_remaining} left")
    cups_remaining -= 1
print("All cups served")
```

`while` loops are powerful but dangerous — if the condition never becomes `False`, the loop runs forever (infinite loop). Always make sure something inside the loop moves toward the exit condition.

---

## `break` and `continue`

```python
# break — stop the loop early
for token in range(1, 11):
    if token == 5:
        print("Sold out!")
        break           # exits immediately
    print(f"Token #{token}")

# continue — skip this iteration
for token in range(1, 11):
    if token % 2 == 0:
        continue        # skip even numbers
    print(f"Odd token: #{token}")
```

---

## `for/else` — Python's Unique Construct

```python
orders = ["masala", "ginger", "elaichi"]
for order in orders:
    if order == "tulsi":
        print("Found tulsi!")
        break
else:
    # This runs ONLY if the loop completed without break
    print("Tulsi not found in orders")
```

---

## Walrus Operator `:=`

Introduced in Python 3.8. Assigns a value inside a condition:

```python
# Without walrus
data = get_next_order()
while data:
    process(data)
    data = get_next_order()

# With walrus — cleaner
while data := get_next_order():
    process(data)
```

---

## Files in This Module (Work Through in Order)

| File | Scenario | What It Teaches |
|------|----------|----------------|
| `01_token_dispneser.py` | Token queue | Basic `for` loop with `range()` |
| `02_batch_chai.py` | Batch brewing | Loops over collections |
| `03_tea_orders.py` | Order list | Iterating with index and value |
| `04_tea_menu.py` | Menu display | Loop over dictionary |
| `05_order_summary.py` | Running total | Accumulator pattern |
| `06_tea-temperature.py` | Temperature monitor | `while` loop |
| `07_put_of_order.py` | Out-of-stock | `break` in a loop |
| `08_for_else.py` | Search pattern | `for/else` construct |
| `09_walrus.py` | Stream processing | Walrus operator `:=` |
| `10_dictionary_case.py` | Dict iteration | `.items()`, `.keys()`, `.values()` |

---

## Run Order

```bash
python3 01_token_dispneser.py
python3 02_batch_chai.py
# ... continue in order through 10_dictionary_case.py
```

---

## Common Mistakes

**Off-by-one errors with `range()`:**
```python
range(10)      # 0 to 9 (not 10!)
range(1, 11)   # 1 to 10
```

**Infinite loops** — always ensure the `while` condition eventually becomes `False`:
```python
count = 0
while count < 5:
    print(count)
    # FORGOT: count += 1  <-- this would loop forever
    count += 1
```

---

## Progress Checklist

- [ ] Can write a `for` loop over `range()` and over a list
- [ ] Can write a `while` loop with a correct exit condition
- [ ] Understand `break` and `continue`
- [ ] Understand `for/else` and when `else` runs
- [ ] Have tried the walrus operator `:=`
- [ ] Know how to iterate over a dictionary with `.items()`
- [ ] Ready to move to Module 05

---

**Next:** [Module 05 — Functions](../05_functions/README.md)
