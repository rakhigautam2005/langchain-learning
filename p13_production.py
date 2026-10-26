import time
import asyncio
from langchain_core.runnables import RunnableLambda

# 1. Retry: a step that fails twice, then works
attempts = {"n": 0}
def flaky(x):
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ValueError("temporary failure")
    return f"ok after {attempts['n']} tries"

retrying = RunnableLambda(flaky).with_retry(stop_after_attempt=4)
print("1) retry:", retrying.invoke("x"))

# 2. Fallback: the main step always fails, the backup answers
def always_fails(x):
    raise RuntimeError("primary down")

safe = RunnableLambda(always_fails).with_fallbacks(
    [RunnableLambda(lambda x: "backup answered: " + x)]
)
print("2) fallback:", safe.invoke("hi"))

# 3. Batch with concurrency vs a plain loop
def slow_double(x):
    time.sleep(1)
    return x * 2

slow = RunnableLambda(slow_double)

start = time.time()
print("3) batch:", slow.batch([1, 2, 3, 4], config={"max_concurrency": 4}))
print("   batch time:", round(time.time() - start, 1), "seconds")

start = time.time()
print("   loop:", [slow.invoke(n) for n in [1, 2, 3, 4]])
print("   loop time:", round(time.time() - start, 1), "seconds")

# 4. Async
async def main():
    return await slow.ainvoke(5)

print("4) async:", asyncio.run(main()))

# 5. Draw a chain (needs: pip install grandalf)
drawn = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
try:
    drawn.get_graph().print_ascii()
except ImportError:
    print("5) run 'pip install grandalf' to draw the graph")