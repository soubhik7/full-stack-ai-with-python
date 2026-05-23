# Module 03 — Conditionals

**Difficulty:** Beginner | **Time:** 2 hours | **Prerequisites:** Module 02

---

## Why This Module Matters

Programs need to make decisions. Without conditionals, code runs the same way every time, regardless of input or state. Conditionals let your program respond differently based on data — the core of any interactive or business application.

Every conditional is an `if` statement that evaluates a **boolean expression** and decides which block of code to run.

---

## Core Concepts

| Concept | Syntax | What It Does |
|---------|--------|-------------|
| `if` | `if condition:` | Runs block only if condition is `True` |
| `elif` | `elif condition:` | Checks next condition if previous was `False` |
| `else` | `else:` | Runs block when all conditions are `False` |
| Comparison operators | `==`, `!=`, `<`, `>`, `<=`, `>=` | Compare two values, returns `bool` |
| Logical operators | `and`, `or`, `not` | Combine multiple conditions |
| Truthy / Falsy | `if value:` | Non-zero numbers, non-empty strings/lists = `True` |
| Nested conditionals | `if` inside `if` | Multi-level decision trees |

---

## How Conditionals Work

```python
kettle_boiled = True
milk_added = False

if kettle_boiled and milk_added:
    print("Ready to brew chai")
elif kettle_boiled:
    print("Kettle ready — add milk")
else:
    print("Start the kettle first")
```

Python evaluates conditions **top to bottom** and executes only the first matching block. Once a block runs, the rest are skipped.

---

## Truthy and Falsy Values

Python treats these as `False`:
- `False`, `None`, `0`, `0.0`, `""` (empty string), `[]`, `{}`, `set()`

Everything else is `True`. This means you can write:

```python
order_list = []
if order_list:
    print("Process orders")
else:
    print("No orders yet")  # This runs
```

---

## Files in This Module (Work Through in Order)

| File | Scenario | What It Teaches |
|------|----------|----------------|
| `01_mini_story_1.py` | Kettle boiling | Basic `if` — simplest possible conditional |
| `02_snak_suggestion.py` | Snack picker | Nested `if/elif/else` chains |
| `03_chai_price_calculator.py` | Pricing logic | Conditions with arithmetic |
| `04_smart_thermostat.py` | Temperature control | Multi-condition chains with `and`/`or` |
| `05_delivery_fees_waiver.py` | Order fees | Business rules with complex conditions |
| `06_train_seat.py` | Seat booking | Combining multiple boolean flags |

---

## Run Order

```bash
python3 01_mini_story_1.py
python3 02_snak_suggestion.py
python3 03_chai_price_calculator.py
python3 04_smart_thermostat.py
python3 05_delivery_fees_waiver.py
python3 06_train_seat.py
```

---

## Common Mistakes

**Mistake 1: Using `=` instead of `==`**
```python
# WRONG — this is assignment, not comparison
if price = 30:

# CORRECT
if price == 30:
```

**Mistake 2: Forgetting the colon**
```python
# WRONG
if price > 50
    print("Expensive")

# CORRECT
if price > 50:
    print("Expensive")
```

**Mistake 3: Inconsistent indentation**
Python uses indentation (4 spaces) to define blocks. Mixing tabs and spaces causes errors.

---

## Exercises to Try

1. Write a conditional that prints "hot", "warm", or "cold" based on a temperature variable
2. Write a chai ordering system that gives a discount if the order is above ₹100 and the customer is a member
3. Add an `else` clause to each file that doesn't have one — what should happen in those cases?

---

## Progress Checklist

- [ ] Can write `if`, `elif`, and `else` correctly
- [ ] Understand truthy and falsy values
- [ ] Can combine conditions with `and` / `or` / `not`
- [ ] Understand nested conditionals
- [ ] Can read a conditional chain and predict the output
- [ ] Ready to move to Module 04

---

**Next:** [Module 04 — Loops](../04_loops/README.md)
