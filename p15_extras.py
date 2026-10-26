from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel

add_one = RunnableLambda(lambda x: x + 1)
double = RunnableLambda(lambda x: x * 2)

# 1. Explicit RunnableSequence: the same thing the | operator builds
seq = RunnableSequence(add_one, double)
print("1) explicit sequence:", seq.invoke(3))

# 2. .pipe(): another way to write |
print("2) pipe:", add_one.pipe(double).invoke(3))

# 3. RunnableParallel with keyword arguments
par = RunnableParallel(plus=add_one, times=double)
print("3) parallel keywords:", par.invoke(5))

# 4. with_config: name and tag a step, useful for tracing
named = add_one.with_config({"run_name": "add_one_step", "tags": ["demo"]})
print("4) with_config:", named.invoke(1))

# 5. map(): apply a Runnable to every item of a list
print("5) map:", add_one.map().invoke([1, 2, 3]))