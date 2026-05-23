# Python Programming — Complete Learning Curriculum

**From Zero to Production-Ready Python Developer**

---

## What This Course Is

A structured, hands-on Python curriculum built around a single theme — a chai (tea) shop — to make abstract programming concepts feel concrete and memorable. Every module uses real business scenarios so you learn *why* a concept exists, not just *how* to use it.

**No prior programming experience required.**

---

## How to Use This Repository

1. Work through modules **in numerical order** — each builds on the previous
2. Read the `README.md` inside each module folder before touching any code
3. Run every file, read the output, then modify it and run again
4. Complete the challenges **only after** finishing the prerequisite modules
5. Track your progress using the table below — mark `[x]` when done

---

## Learning Path & Progress Tracker

| # | Module | What You Learn | Difficulty | Est. Time | Done |
|---|--------|---------------|------------|-----------|------|
| 00 | [Python Environment](00_python/) | Setup, syntax basics, Python version | Beginner | 30 min | [ ] |
| 01 | *(Virtual Environments)* | `venv`, pip, package isolation | Beginner | 30 min | [ ] |
| 02 | [Data Types](02_datatypes/) | Variables, strings, numbers, lists, dicts, sets | Beginner | 3–4 hrs | [ ] |
| 03 | [Conditionals](03_conditionals/) | if/elif/else, decision logic, boolean operators | Beginner | 2 hrs | [ ] |
| 04 | [Loops](04_loops/) | for, while, break, continue, walrus operator | Beginner | 2–3 hrs | [ ] |
| 05 | [Functions](05_functions/) | Defining functions, scope, parameters, return | Intermediate | 3 hrs | [ ] |
| 06 | [Chai Business App](06_chai_business/) | Modules, packages, project structure | Intermediate | 2 hrs | [ ] |
| 07 | [Comprehensions](07_comprehensions/) | List/set/dict comprehensions, expressions | Intermediate | 2 hrs | [ ] |
| 08 | [Generators](08_generators/) | yield, lazy evaluation, memory efficiency | Intermediate | 2 hrs | [ ] |
| 09 | [Decorators](09_decorators/) | Function wrapping, @syntax, real-world patterns | Intermediate | 2 hrs | [ ] |
| 10 | [OOP](10_oop/) | Classes, inheritance, MRO, properties, classmethods | Advanced | 4–5 hrs | [ ] |
| 11 | [Exceptions](11_exceptions/) | try/except, custom exceptions, file handling | Intermediate | 2–3 hrs | [ ] |
| 12 | [Threading & Multiprocessing](12_threads_concurrency/) | Threads, processes, GIL, locks, queues | Advanced | 3 hrs | [ ] |
| 13 | [Async Python](13_async_python/) | asyncio, async/await, race conditions, deadlocks | Advanced | 3 hrs | [ ] |
| 14 | [Pydantic](14_pydantic/) | Data validation, type enforcement, serialization | Advanced | 2–3 hrs | [ ] |

### Challenges (Apply Everything)

| # | Project | Modules Required | Difficulty | Done |
|---|---------|-----------------|------------|------|
| C1 | [Utility Scripts](challenges/01_utilities/) | 00–05 | Beginner | [ ] |
| C2 | [Data Handling](challenges/02_data_handling/) | 02, 05, 11 | Intermediate | [ ] |
| C3 | [Web Scraping](challenges/03_web_scraping/) | 05, 11 + `requests`/`bs4` | Intermediate | [ ] |
| C4 | [Automation](challenges/04_automation/) | 04, 05, 11 | Intermediate | [ ] |
| C5 | [Data Science](challenges/05_data_science/) | 02, 05, 07 + `pandas`/`matplotlib` | Advanced | [ ] |
| C6 | [URL Shortener (Flask App)](challenges/06_url_shortner/) | 05, 10, 11 + `Flask`/`SQLite` | Advanced | [ ] |

---

## Curriculum Map

```
BEGINNER
  00 Environment Setup
       |
  02 Data Types ──── strings, numbers, lists, tuples, dicts, sets
       |
  03 Conditionals ── if/elif/else, boolean logic
       |
  04 Loops ───────── for, while, break, continue, walrus

INTERMEDIATE
       |
  05 Functions ───── parameters, scope, return, closures
       |
  06 App Structure ─ modules, packages (first real project)
       |
  07 Comprehensions ─ elegant one-liners
       |
  08 Generators ──── yield, lazy iteration
       |
  09 Decorators ──── @syntax, function wrapping
       |
  11 Exceptions ──── try/except, custom errors

ADVANCED
       |
  10 OOP ─────────── classes, inheritance, MRO, properties
       |
  12 Concurrency ─── threads, processes, GIL, locks
       |
  13 Async ─────────── asyncio, async/await, race conditions
       |
  14 Pydantic ──────── validation, type safety

PROJECTS (use what you've learned)
  C1 → C2 → C3 → C4 → C5 → C6
```

---

## Module Summaries

### Module 00 — Python Environment
Verify Python is installed and understand the runtime. Short and sweet — just enough to get you moving.

### Module 02 — Data Types
The foundation. Python stores *everything* as an object with a type. This module teaches you what types exist, how Python handles memory (using `id()`), and how to manipulate each type.

### Module 03 — Conditionals
Programs make decisions. Conditionals let your code branch based on data — critical for any real application. Built around a chai shop pricing scenario.

### Module 04 — Loops
Repeating logic without repeating code. Learn `for`, `while`, `break`, `continue`, `for/else`, and the walrus operator (`:=`). Every program needs loops.

### Module 05 — Functions
The most important module. Functions are how you organize and reuse logic. Understanding scope, parameters, closures, and return values is non-negotiable for writing real software.

### Module 06 — Application Structure
Real projects are not single files. Learn how Python's module and package system works by building a structured chai business application.

### Module 07 — Comprehensions
Python's most loved feature. Write data transformation in one clean line instead of five messy ones. List, set, and dictionary comprehensions, plus generator expressions.

### Module 08 — Generators
Large datasets don't fit in memory. Generators produce values one at a time using `yield`, letting you process infinite sequences efficiently. Foundational for data pipelines.

### Module 09 — Decorators
Decorators wrap functions to add behavior (logging, auth, timing) without touching the function's code. Built on closures from Module 05. Used everywhere in production Python.

### Module 10 — Object-Oriented Programming
Model real-world entities as objects with state (attributes) and behavior (methods). Covers classes, `__init__`, inheritance, MRO, `@staticmethod`, `@classmethod`, and `@property`.

### Module 11 — Exceptions
Production code fails gracefully. Learn to handle errors with `try/except`, write custom exception classes, and manage file operations safely.

### Module 12 — Threading & Multiprocessing
Run multiple tasks simultaneously. Understand Python's GIL, when to use threads vs. processes, how to protect shared data with locks, and inter-process communication with queues.

### Module 13 — Async Python
The modern concurrency model. `asyncio` with `async/await` lets a single thread handle thousands of I/O operations concurrently. Also covers race conditions and deadlocks.

### Module 14 — Pydantic
Data coming from APIs, databases, and users is untrusted. Pydantic validates and parses data against type annotations, catching errors early. Standard in FastAPI and modern Python services.

---

## Setup

```bash
# 1. Install Python 3.10+
python3 --version

# 2. Create a virtual environment (do this once per project)
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# 3. Run any file
python3 02_datatypes/chapter_1.py

# 4. Install challenge dependencies
pip install requests beautifulsoup4 pandas matplotlib flask pydantic
```

---

## Technologies Used

| Technology | Used In |
|-----------|---------|
| Python 3.10+ | All modules |
| Flask | Challenge 6 (URL Shortener) |
| SQLite | Challenge 6 |
| Pandas | Challenge 5 (Data Science) |
| Matplotlib / Seaborn | Challenge 5 |
| Requests | Challenge 3 (Web Scraping) |
| BeautifulSoup4 | Challenge 3 |
| Pydantic | Module 14 |
| asyncio | Module 13 |
| threading / multiprocessing | Module 12 |

---

## Tips for Effective Learning

- **Run the code** — reading without running is not learning
- **Break things intentionally** — comment out a line, change a value, see what error you get
- **Type it out** — don't copy-paste; muscle memory matters
- **One module per day** is a solid pace for beginners
- **Challenge yourself** — once a module clicks, try extending the examples

---

## Repository Structure

```
01_python/
├── readme.md                    ← You are here
├── 00_python/                   ← Module 00: Environment
├── 02_datatypes/                ← Module 02: Data Types
├── 03_conditionals/             ← Module 03: Conditionals
├── 04_loops/                    ← Module 04: Loops
├── 05_functions/                ← Module 05: Functions
├── 06_chai_business/            ← Module 06: App Structure
├── 07_comprehensions/           ← Module 07: Comprehensions
├── 08_generators/               ← Module 08: Generators
├── 09_decorators/               ← Module 09: Decorators
├── 10_oop/                      ← Module 10: OOP
├── 11_exceptions/               ← Module 11: Exceptions
├── 12_threads_concurrency/      ← Module 12: Threading
├── 13_async_python/             ← Module 13: Async
├── 14_pydantic/                 ← Module 14: Pydantic
└── challenges/                  ← 6 hands-on projects
    ├── 01_utilities/
    ├── 02_data_handling/
    ├── 03_web_scraping/
    ├── 04_automation/
    ├── 05_data_science/
    └── 06_url_shortner/
```

---

*Start with `00_python/README.md` and work forward. Each folder has its own README with detailed instructions.*
