# Module 00 — Python Environment & Syntax Basics

**Difficulty:** Beginner | **Time:** 30 minutes | **Prerequisites:** None

---

## Why This Module Exists

Before writing a single line of real Python, you need to know:
- Is Python installed correctly on your machine?
- What version are you running?
- What does a basic Python script look like?

This module answers those three questions. It's intentionally short.

---

## Key Concepts

| Concept | What It Is |
|---------|-----------|
| `sys.version` | A string containing the Python version and build info |
| `import` | Brings an external module into your script |
| `print()` | Outputs text to the terminal |

---

## Files in This Module

| File | What It Teaches |
|------|----------------|
| `testpython.py` | Verifying Python works; reading version info via `sys` |
| `non_python_code.py` | See what non-Pythonic code looks like vs clean Python |
| `non_python_shop.py` | A shop scenario written without Python idioms |

---

## Run Order

```bash
python3 testpython.py
python3 non_python_code.py
python3 non_python_shop.py
```

---

## What to Look For

When you run `testpython.py`, you should see something like:
```
3.11.4 (main, Jul 5 2023, ...) [Clang ...]
```

The first number (`3`) is the major version. This course requires **Python 3.10 or higher**.

---

## Why `sys.version`?

The `sys` module is part of Python's standard library — it gives you access to interpreter details. Checking the version this way is a common first step in scripts that need to enforce a minimum Python version.

---

## Progress Checklist

- [ ] Ran `testpython.py` and saw a valid Python 3.x version
- [ ] Understand what `import` does
- [ ] Understand what `print()` does
- [ ] Ready to move to Module 02

---

**Next:** [Module 02 — Data Types](../02_datatypes/README.md)
