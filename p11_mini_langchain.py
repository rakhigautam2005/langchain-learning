from abc import ABC, abstractmethod
import random

# ---------- The standard plug: every component must have invoke() ----------
class MyRunnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass

# ---------- Components that follow the standard ----------
class MyPrompt(MyRunnable):
    def __init__(self, template):
        self.template = template
    def invoke(self, input_data):
        return self.template.format(**input_data)

class MyFakeLLM(MyRunnable):
    def invoke(self, prompt):
        replies = ["It is a framework.", "It connects models with tools."]
        return {"response": random.choice(replies)}

class MyParser(MyRunnable):
    def invoke(self, input_data):
        return input_data["response"]

class MyShout(MyRunnable):
    def invoke(self, text):
        return text.upper()

# ---------- NEW: your counter, a brand new component ----------
class MyCounter(MyRunnable):
    def invoke(self, text):
        return len(text.split())

# ---------- The connector: runs steps in order ----------
class MyChain(MyRunnable):
    def __init__(self, steps):
        self.steps = steps
    def invoke(self, input_data):
        for step in self.steps:
            input_data = step.invoke(input_data)
        return input_data

# ---------- Part 1: a simple chain ----------
explain = MyChain([MyPrompt("Explain {topic} in one line"), MyFakeLLM(), MyParser()])
print("1) simple chain:", explain.invoke({"topic": "LangChain"}))

# ---------- Part 2: a chain inside a chain ----------
bigger = MyChain([explain, MyShout()])
print("2) chain inside a chain:", bigger.invoke({"topic": "LangChain"}))

# ---------- NEW: Part 4, plug the counter in with no glue code ----------
counted = MyChain([explain, MyCounter()])
print("4) word count:", counted.invoke({"topic": "LangChain"}))

# ---------- Part 3: the OLD world, a different method name on every class ----------
class OldPrompt:
    def format_it(self, d):
        return "Explain {topic}".format(**d)

class OldLLM:
    def predict(self, p):
        return "It is a framework."

class OldParser:
    def parse(self, t):
        return t.upper()

p = OldPrompt().format_it({"topic": "LangChain"})
r = OldLLM().predict(p)
print("3) old way, manual glue:", OldParser().parse(r))

try:
    MyChain([OldPrompt(), OldLLM(), OldParser()]).invoke({"topic": "LangChain"})
except AttributeError as e:
    print("3) generic chain fails:", e)