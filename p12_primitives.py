from operator import itemgetter
from langchain_core.runnables import (
    RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch, chain
)

add_one = RunnableLambda(lambda x: x + 1)
double = RunnableLambda(lambda x: x * 2)

# 1. Sequence: one step after another
seq = add_one | double
print("1) sequence:", type(seq).__name__, seq.invoke(3))

# 2. Parallel: same input to every branch, result is a dict
par = RunnableParallel({"plus": add_one, "times": double, "same": RunnablePassthrough()})
print("2) parallel:", par.invoke(5))

# 3. A plain dict inside a chain becomes a RunnableParallel
shortcut = add_one | {"a": double, "b": RunnableLambda(lambda x: x - 1)}
print("3) dict shortcut:", shortcut.invoke(5))

# 4. assign keeps the input dict and adds a key
assigned = RunnablePassthrough.assign(total=lambda d: d["price"] * d["qty"])
print("4) assign:", assigned.invoke({"price": 10, "qty": 3}))

# 5. itemgetter picks one key out of a dict
price_only = RunnableLambda(lambda d: d) | itemgetter("price")
print("5) itemgetter:", price_only.invoke({"price": 10, "qty": 3}))

# 6. Branch: if / elif / else
size = RunnableBranch(
    (lambda x: x < 0, RunnableLambda(lambda x: "negative")),
    (lambda x: x == 0, RunnableLambda(lambda x: "zero")),
    RunnableLambda(lambda x: "positive"),
)
print("6) branch:", [size.invoke(n) for n in (-5, 0, 9)])

# 7. Dynamic routing: a function that returns which Runnable to run
def route(x):
    if isinstance(x, str):
        return RunnableLambda(lambda s: s.upper())
    return RunnableLambda(lambda n: n * 100)

router = RunnableLambda(route)
print("7) router:", router.invoke("hello"), router.invoke(3))

# 8. The @chain decorator turns a function into a Runnable
@chain
def shout_twice(text):
    return (text + "! ") * 2

print("8) @chain:", shout_twice.invoke("hey"))

# 9. A full pipeline built from fake parts (no API calls)
fake_prompt = RunnableLambda(lambda d: f"Explain {d['topic']} in one line")
fake_llm = RunnableLambda(lambda p: f"[model answer to: {p}]")
fake_parser = RunnableLambda(lambda s: s.strip("[]"))
pipeline = fake_prompt | fake_llm | fake_parser
print("9) pipeline:", pipeline.invoke({"topic": "Runnables"}))