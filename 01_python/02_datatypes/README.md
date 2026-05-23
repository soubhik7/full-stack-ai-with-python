# Module 02 — Data Types

**Difficulty:** Beginner | **Time:** 3–4 hours | **Prerequisites:** Module 00

---

## Why This Module Matters

Everything in Python is an object, and every object has a **type**. Before you can write logic, loop over data, or build functions, you need to know:
- What types of data Python can store
- How Python stores data in memory (`id()`)
- How to manipulate each type

This is the most foundational module in the course. Spend extra time here.

---

## Core Concepts

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| Variable | A named reference to a value in memory | All programs store and manipulate data |
| `id()` | Returns the memory address of an object | Explains why variables point, not copy |
| String (`str`) | Text data in quotes | Every program handles text |
| Integer (`int`) | Whole numbers | Counting, indexing, arithmetic |
| Float (`float`) | Decimal numbers | Prices, measurements, percentages |
| Boolean (`bool`) | `True` or `False` | Decision-making, flags |
| List (`list`) | Ordered, mutable collection | Most common collection type |
| Tuple (`tuple`) | Ordered, **immutable** collection | Safe, unchangeable sequences |
| Dictionary (`dict`) | Key-value pairs | Fast lookups by name |
| Set (`set`) | Unordered, **unique** values | Deduplication, membership testing |
| Type conversion | Changing one type to another | Parsing user input, API responses |

---

## The `id()` Function — Why It Matters

Python doesn't copy values when you assign a variable. It creates a **reference** to an object in memory.

```python
a = 14
b = a
# Both a and b point to the SAME memory location
print(id(a) == id(b))  # True

b = 99
# Now b points to a NEW location; a is unchanged
print(a)  # Still 14
```

Understanding this prevents a whole category of bugs — especially with mutable types like lists.

---

## Files in This Module (Work Through in Order)

| File | Topic | Key Concept |
|------|-------|-------------|
| `chapter_1.py` | Variables & memory | `id()`, reassignment, mutable vs immutable |
| `chapter_2.py` | Sets | `set.add()`, unique values, mutability paradox |
| `chapter_3.py` | Integers & arithmetic | `+`, `-`, `*`, `/`, `//`, `%`, `**`, `_` separator |
| `chapter_4.py` | Booleans | `True`/`False`, upcasting, `and`/`or` operators |
| `chapter_5.py` | Lists | Indexing, `append`, `remove`, `pop`, slicing |
| `chapter_6.py` | Tuples | Immutability, use cases, tuple unpacking |
| `chapter_7.py` | Dictionaries | `dict[key]`, `.get()`, `.keys()`, `.values()` |
| `chapter_8.py` | Sets (advanced) | Set operations: union, intersection, difference |
| `chapter_9.py` | Type conversion | `int()`, `str()`, `float()`, `bool()` |
| `chapter_10.py` | String methods | `.upper()`, `.split()`, `.strip()`, f-strings |
| `chapter_11.py` | Advanced operations | Nested data, combined type usage |

---

## The Mutable vs Immutable Distinction

This is the most important concept in this module:

| Mutable (can change) | Immutable (cannot change) |
|---------------------|--------------------------|
| `list` | `str` |
| `dict` | `int`, `float`, `bool` |
| `set` | `tuple` |

When you "modify" an immutable object, Python actually creates a new one.
When you modify a mutable object, the change happens in-place — affecting all references.

---

## Run Order

```bash
python3 chapter_1.py
python3 chapter_2.py
# ... continue through chapter_11.py
```

---

## Exercises to Try

1. Create a list of chai flavors, add two more, remove one, then print the final list
2. Create a dictionary of tea prices, update one price, and print only the keys
3. Try adding a duplicate to a set — notice what happens
4. Convert the string `"42"` to an integer and add 8 to it

---

## Progress Checklist

- [ ] Understand the difference between mutable and immutable types
- [ ] Can explain what `id()` returns and why it matters
- [ ] Comfortable with list indexing and slicing
- [ ] Know when to use a dict vs a list vs a set
- [ ] Understand type conversion (casting)
- [ ] Ready to move to Module 03

---

**Next:** [Module 03 — Conditionals](../03_conditionals/README.md)
