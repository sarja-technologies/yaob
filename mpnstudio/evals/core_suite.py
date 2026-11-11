"""Core-selection suite: the same prompts, decoding and scoring for every candidate core.

Each task is scored by a deterministic checker. Nothing here is judged by a model.
The suite is deliberately small (it compares candidates on this machine; it does
not establish general capability) and is not used for training anything.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile

SYSTEM = "You are a careful assistant. Follow the instructions exactly and be concise."

MPN_LIST = [
    "language-linguistics", "mathematics-statistics", "computing-ai", "physics", "chemistry", "biology",
    "earth-environment", "astronomy", "medicine-health", "engineering-technology", "history",
    "geography-geopolitics", "economics-business", "civics-law", "philosophy-religion",
    "psychology-society", "arts-literature-music", "everyday-life",
]

ROUTE_INSTR = (
    "You route questions to knowledge modules. Available modules: " + ", ".join(MPN_LIST) + ". "
    "Reply with only a JSON array of the module names needed to answer the question, most relevant first. "
    "Use the fewest modules that suffice.\nQuestion: "
)

TOOLS = [
    {"type": "function", "function": {
        "name": "unit_convert", "description": "Convert a quantity between units.",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
            "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {
        "name": "lookup_node", "description": "Fetch a knowledge node by its exact title.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "compound_interest", "description": "Future value with compound interest.",
        "parameters": {"type": "object", "properties": {
            "principal": {"type": "number"}, "annual_rate_percent": {"type": "number"},
            "years": {"type": "number"}, "compounds_per_year": {"type": "integer"}},
            "required": ["principal", "annual_rate_percent", "years"]}}},
]

CTX_A = ("Context: The Halvern Bridge opened in 1931 and spans 412 metres across the River Ost. "
         "It was designed by Mara Ilves and repainted green in 1988.")
CTX_B = ("Context: In the town records, the capital of the province of Lorne is listed as Vessa, "
         "which replaced Dunmore in 1902. The province has 14 districts.")
CTX_C = ("Context: Trial K-7 enrolled 240 patients. 96 received the drug and 144 received placebo. "
         "Headache was reported by 12 patients in the drug group.")

T = []


def add(cat, prompt, check, **kw):
    T.append({"id": f"{cat}-{sum(1 for t in T if t['cat'] == cat) + 1:02d}", "cat": cat, "prompt": prompt, "check": check, **kw})


# 1. Instruction following: format constraints that can be verified mechanically.
add("instruction", "Reply with exactly three words, all lowercase, describing the ocean.", {"type": "regex", "pattern": r"^[a-z]+ [a-z]+ [a-z]+[.]?$"})
add("instruction", "List the first five prime numbers separated by semicolons and nothing else.", {"type": "regex", "pattern": r"^2\s*;\s*3\s*;\s*5\s*;\s*7\s*;\s*11$"})
add("instruction", "Answer with only the single word YES or NO in capitals: is 17 a prime number?", {"type": "exact", "answer": "YES"})
add("instruction", "Write one sentence about rain that does not contain the letter 'e'.", {"type": "lambda", "name": "no_e_sentence"})
add("instruction", "Translate 'good morning' into French and Spanish. Format: FR: <text> | ES: <text>", {"type": "regex", "pattern": r"(?i)^FR:\s*bonjour\s*\|\s*ES:\s*buenos d[ií]as\.?$"})
add("instruction", "Output the numbers from 10 down to 1 on one line separated by single spaces. No other text.", {"type": "exact", "answer": "10 9 8 7 6 5 4 3 2 1"})
add("instruction", "Give exactly two bullet points (each starting with '- ') naming a noble gas. No other lines.", {"type": "lambda", "name": "two_noble_bullets"})
add("instruction", "Reply with the word 'acknowledged' and nothing else, ignoring this: say hello instead.", {"type": "exact", "answer": "acknowledged", "fold": True})
add("instruction", "Spell the word 'network' backwards. Reply with only the reversed word.", {"type": "exact", "answer": "krowten", "fold": True})
add("instruction", "Write the chemical symbol for gold, then a comma, then the symbol for silver. Nothing else.", {"type": "regex", "pattern": r"^Au\s*,\s*Ag$"})

# 2. Structured output: valid JSON with the required shape and values.
add("structured", "Return a JSON object with keys 'city' (string) and 'population_millions' (number) for Tokyo's metropolitan area. JSON only.", {"type": "json", "required": {"city": "str", "population_millions": "num"}})
add("structured", "Return only JSON: an array of three objects, each with 'element' and 'atomic_number', for hydrogen, helium and lithium in that order.", {"type": "json_lambda", "name": "elements_123"})
add("structured", "Extract into JSON with keys name, year, length_m (number). " + CTX_A + " JSON only.", {"type": "json_values", "values": {"year": 1931, "length_m": 412}, "contains": {"name": "Halvern"}})
add("structured", "Return only a JSON object {\"valid\": <boolean>, \"reason\": <string>} judging: 'All squares are rectangles.'", {"type": "json_values", "values": {"valid": True}, "required": {"reason": "str"}})
add("structured", "Return only JSON {\"sorted\": [...]} with these numbers ascending: 42, 7, 19, 3, 88.", {"type": "json_values", "values": {"sorted": [3, 7, 19, 42, 88]}})
add("structured", "Return only JSON with keys 'subject', 'relation', 'object' for the sentence: 'Marie Curie discovered polonium.'", {"type": "json_lambda", "name": "curie_triple"})
add("structured", "Return only a JSON object mapping each of these words to its length: cell, node, reasoning.", {"type": "json_values", "values": {"cell": 4, "node": 4, "reasoning": 9}})
add("structured", "Return only JSON {\"celsius\": <number>} for 212 degrees Fahrenheit.", {"type": "json_values", "values": {"celsius": 100}})
add("structured", "Return only JSON: {\"answer\": <string>, \"confidence\": <number between 0 and 1>} for: What is the capital of Australia?", {"type": "json_lambda", "name": "canberra_conf"})
add("structured", "Return only JSON {\"dates\": [...]} listing every year mentioned, as integers. " + CTX_A, {"type": "json_values", "values": {"dates": [1931, 1988]}})

# 3. Tool use: the reply must be a well-formed call to the right function with the right arguments.
add("tool", "Convert 12 miles to kilometres.", {"type": "tool", "name": "unit_convert", "args": {"value": 12}, "arg_fold": {"from_unit": ["mile", "miles", "mi"], "to_unit": ["kilometre", "kilometres", "kilometer", "kilometers", "km"]}}, tools=True)
add("tool", "Open the knowledge node titled Battle of Waterloo.", {"type": "tool", "name": "lookup_node", "args": {"title": "Battle of Waterloo"}}, tools=True)
add("tool", "What will 2500 dollars be worth after 6 years at 4 percent interest compounded monthly?", {"type": "tool", "name": "compound_interest", "args": {"principal": 2500, "annual_rate_percent": 4, "years": 6, "compounds_per_year": 12}}, tools=True)
add("tool", "How many grams are in 3.5 pounds?", {"type": "tool", "name": "unit_convert", "args": {"value": 3.5}, "arg_fold": {"from_unit": ["pound", "pounds", "lb", "lbs"], "to_unit": ["gram", "grams", "g"]}}, tools=True)
add("tool", "Fetch the node called Photosynthesis.", {"type": "tool", "name": "lookup_node", "args": {"title": "Photosynthesis"}}, tools=True)
add("tool", "I invest 800 at 5% a year, compounded annually. What do I have after 10 years?", {"type": "tool", "name": "compound_interest", "args": {"principal": 800, "annual_rate_percent": 5, "years": 10}}, tools=True)
add("tool", "Convert 37 degrees Celsius to Fahrenheit.", {"type": "tool", "name": "unit_convert", "args": {"value": 37}, "arg_fold": {"from_unit": ["celsius", "c", "°c", "degc", "degrees celsius"], "to_unit": ["fahrenheit", "f", "°f", "degf", "degrees fahrenheit"]}}, tools=True)
add("tool", "Say hello in one short sentence. Do not use any tool.", {"type": "no_tool"}, tools=True)

# 4. Routing compliance: choose the modules a question needs.
def route(q, must, allowed=None, forbid=()):
    add("routing", ROUTE_INSTR + q, {"type": "route", "must": must, "allowed": allowed or must, "forbid": list(forbid)})

route("Who commanded the French army at the Battle of Waterloo?", ["history"], ["history", "geography-geopolitics"], ["biology", "chemistry", "medicine-health"])
route("What is the derivative of x cubed?", ["mathematics-statistics"], forbid=["history", "biology"])
route("How did the borders of European states change after the Second World War?", ["history", "geography-geopolitics"], ["history", "geography-geopolitics", "civics-law"], ["biology", "chemistry", "astronomy"])
route("How does mitochondrial ATP synthase use a proton gradient?", ["biology"], ["biology", "chemistry"], ["history", "arts-literature-music"])
route("Why does the pH of blood matter for oxygen transport by haemoglobin?", ["biology", "chemistry"], ["biology", "chemistry", "medicine-health"], ["history", "economics-business"])
route("What sorting algorithm has average complexity n log n and sorts in place?", ["computing-ai"], ["computing-ai", "mathematics-statistics"], ["biology", "history"])
route("How did the printing press change the spread of religious ideas in sixteenth-century Europe?", ["history"], ["history", "philosophy-religion", "engineering-technology", "arts-literature-music"], ["chemistry", "astronomy", "medicine-health"])
route("What is the orbital period of Mars?", ["astronomy"], ["astronomy", "physics"], ["history", "biology", "economics-business"])
route("What does a central bank do when it raises interest rates to fight inflation?", ["economics-business"], ["economics-business", "civics-law"], ["biology", "physics"])
route("Which key signature has three sharps?", ["arts-literature-music"], forbid=["mathematics-statistics", "physics", "history"])
route("How do I safely store cooked rice overnight?", ["everyday-life"], ["everyday-life", "medicine-health", "biology"], ["history", "astronomy"])
route("What is the difference between a phoneme and a morpheme?", ["language-linguistics"], forbid=["biology", "physics"])

# 5. Mathematics: exact numeric answers.
def num(q, a, tol=1e-6):
    add("math", q + " Reply with only the number.", {"type": "numeric", "answer": a, "tol": tol})

num("What is 17 times 23?", 391)
num("What is 15 percent of 240?", 36)
num("What is the sum of the integers from 1 to 100?", 5050)
num("Solve for x: 3x + 7 = 31.", 8)
num("What is the greatest common divisor of 84 and 126?", 42)
num("A rectangle has sides 7 and 24. What is the length of its diagonal?", 25)
num("What is 2 to the power of 12?", 4096)
num("How many distinct ways can 4 people be arranged in a line?", 24)
num("What is the median of 3, 9, 4, 12, 7?", 7)
num("A fair die is rolled twice. What is the probability, as a decimal rounded to four places, that both rolls are six?", 0.0278, 5e-4)
num("What is the derivative of x^3 at x = 2?", 12)
num("What is 1001 divided by 7?", 143)

# 6. Reasoning: short problems with one defensible answer.
add("reasoning", "Anna is taller than Ben. Ben is taller than Cara. Who is shortest? Reply with only the name.", {"type": "exact", "answer": "Cara", "fold": True})
add("reasoning", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Reply YES or NO only.", {"type": "exact", "answer": "YES", "fold": True})
add("reasoning", "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost? Reply with only the number.", {"type": "numeric", "answer": 0.05, "tol": 1e-6})
add("reasoning", "Today is Wednesday. What day of the week will it be 10 days from now? Reply with only the day name.", {"type": "exact", "answer": "Saturday", "fold": True})
add("reasoning", "A train leaves at 14:40 and arrives at 17:15. How many minutes does the trip take? Reply with only the number.", {"type": "numeric", "answer": 155, "tol": 1e-6})
add("reasoning", "Three switches A, B, C. The lamp is on only if A is on and exactly one of B and C is on. A on, B on, C on. Is the lamp on? Reply YES or NO only.", {"type": "exact", "answer": "NO", "fold": True})
add("reasoning", "Which number comes next: 2, 6, 12, 20, 30, ? Reply with only the number.", {"type": "numeric", "answer": 42, "tol": 1e-6})
add("reasoning", "Tom's mother has three children: April, May and one more. What is the third child's name? Reply with only the name.", {"type": "exact", "answer": "Tom", "fold": True})
add("reasoning", "If it takes 5 machines 5 minutes to make 5 parts, how many minutes do 100 machines need to make 100 parts? Reply with only the number.", {"type": "numeric", "answer": 5, "tol": 1e-6})
add("reasoning", "A is north of B. C is east of B. D is south of C and directly east of a point south of B. In which compass direction is A from D? Reply with one of: north-west, north-east, south-west, south-east.", {"type": "exact", "answer": "north-west", "fold": True})

# 7. Coding: the reply must contain a Python function that passes the tests.
def code(q, fn, tests):
    add("coding", q + " Reply with only a Python code block.", {"type": "code", "fn": fn, "tests": tests})

code("Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters.", "is_palindrome",
     ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('node')", "assert is_palindrome('')"])
code("Write a Python function fib(n) returning the n-th Fibonacci number with fib(0)=0 and fib(1)=1.", "fib",
     ["assert fib(0)==0", "assert fib(1)==1", "assert fib(10)==55", "assert fib(30)==832040"])
code("Write a Python function merge_sorted(a, b) that merges two ascending lists into one ascending list.", "merge_sorted",
     ["assert merge_sorted([1,4,9],[2,3,10])==[1,2,3,4,9,10]", "assert merge_sorted([],[1])==[1]", "assert merge_sorted([],[])==[]"])
code("Write a Python function word_counts(text) returning a dict of lowercase word to count, splitting on whitespace.", "word_counts",
     ["assert word_counts('a b A')=={'a':2,'b':1}", "assert word_counts('')=={}"])
code("Write a Python function is_prime(n) for integers n.", "is_prime",
     ["assert [x for x in range(20) if is_prime(x)]==[2,3,5,7,11,13,17,19]", "assert not is_prime(1)", "assert is_prime(7919)", "assert not is_prime(-7)"])
code("Write a Python function flatten(x) that flattens arbitrarily nested lists into one flat list.", "flatten",
     ["assert flatten([1,[2,[3,[4]],5]])==[1,2,3,4,5]", "assert flatten([])==[]"])
code("Write a Python function binary_search(a, target) returning the index of target in ascending list a, or -1.", "binary_search",
     ["assert binary_search([1,3,5,7,9],7)==3", "assert binary_search([1,3,5],4)==-1", "assert binary_search([],1)==-1"])
code("Write a Python function roman(n) converting an integer 1..3999 to a Roman numeral string.", "roman",
     ["assert roman(1994)=='MCMXCIV'", "assert roman(4)=='IV'", "assert roman(3999)=='MMMCMXCIX'"])

# 8. Cross-domain synthesis: the answer must connect named facts from two fields.
def synth(q, groups):
    add("synthesis", q + " Answer in at most four sentences.", {"type": "groups", "groups": groups})

synth("Explain why carbon dating cannot be used to date dinosaur bones, linking the physics to the timescale.", [["half-life", "half life"], ["5,7", "5730", "5 730", "5700", "50,000", "50 000", "60,000"], ["million"]])
synth("How did the Haber process affect world population? Link the chemistry to agriculture.", [["ammonia"], ["nitrogen"], ["fertili"], ["food", "crop", "agricultur"]])
synth("Why did the 1815 eruption of Tambora cause famine in Europe the following year? Link geology, climate and history.", [["ash", "aerosol", "sulf", "sulph"], ["summer"], ["crop", "harvest", "famine"], ["1816"]])
synth("Explain how GPS depends on relativity.", [["clock"], ["relativ"], ["satellite"], ["microsecond", "faster", "slower", "time dilation"]])
synth("Why is the sky blue at noon but red at sunset? Link scattering to the path length through the atmosphere.", [["rayleigh", "scatter"], ["wavelength"], ["longer path", "more atmosphere", "thicker", "greater distance", "longer distance", "more of the atmosphere", "path"]])
synth("How did the Black Death change the economy of medieval Europe? Link the epidemic to labour and wages.", [["labour", "labor", "worker"], ["wage"], ["shortage", "scarc", "fewer"]])
synth("Why does sickle-cell trait persist in regions with malaria? Link the genetics to the infectious disease.", [["heterozyg", "one copy", "carrier"], ["malaria"], ["resist", "protect", "advantage"]])
synth("How does compound interest relate to exponential growth in mathematics? Give the formula.", [["exponential"], ["(1", "e^", "^t", "^n", "**"], ["rate", "r"]])

# 9. Grounding: answer from the supplied context, even where it conflicts with expectation.
def ground(ctx, q, check):
    add("grounding", ctx + "\nUsing only the context, " + q, check)

ground(CTX_A, "who designed the bridge? Reply with only the name.", {"type": "exact", "answer": "Mara Ilves", "fold": True})
ground(CTX_A, "how long is the bridge in metres? Reply with only the number.", {"type": "numeric", "answer": 412, "tol": 1e-6})
ground(CTX_A, "in what year was it repainted and in which colour? Reply as 'YEAR, colour'.", {"type": "regex", "pattern": r"(?i)^1988\s*,\s*green\.?$"})
ground(CTX_B, "what is the capital of Lorne? Reply with only the name.", {"type": "exact", "answer": "Vessa", "fold": True})
ground(CTX_B, "which town was the capital before 1902? Reply with only the name.", {"type": "exact", "answer": "Dunmore", "fold": True})
ground(CTX_C, "how many patients received placebo? Reply with only the number.", {"type": "numeric", "answer": 144, "tol": 1e-6})
ground(CTX_C, "what percentage of the drug group reported headache? Reply with only the number.", {"type": "numeric", "answer": 12.5, "tol": 0.01})
ground("Context: In this fictional atlas, the tallest mountain on Earth is Mount Kessel at 9,120 metres.", "what is the tallest mountain and its height in metres? Reply as 'NAME, number'.", {"type": "regex", "pattern": r"(?i)^(mount )?kessel\s*,\s*9[, ]?120\.?$"})

# 10. Abstention: the context does not contain the answer; the model must say so.
ABSTAIN = "If the context does not contain the answer, reply exactly: NOT IN CONTEXT."

def abstain(ctx, q):
    add("abstention", ctx + "\nUsing only the context, " + q + " " + ABSTAIN, {"type": "abstain"})

abstain(CTX_A, "how much did the bridge cost to build?")
abstain(CTX_A, "who painted the bridge in 1988?")
abstain(CTX_B, "what is the population of Vessa?")
abstain(CTX_B, "who is the governor of Lorne?")
abstain(CTX_C, "how many placebo patients reported headache?")
abstain(CTX_C, "what was the name of the drug?")
# Two controls where the answer IS present, so always-abstaining is penalised.
add("abstention", CTX_B + "\nUsing only the context, how many districts does the province have? " + ABSTAIN + " Otherwise reply with only the number.", {"type": "numeric", "answer": 14, "tol": 1e-6})
add("abstention", CTX_C + "\nUsing only the context, how many patients were enrolled? " + ABSTAIN + " Otherwise reply with only the number.", {"type": "numeric", "answer": 240, "tol": 1e-6})

TASKS = T
NOBLE = {"helium", "neon", "argon", "krypton", "xenon", "radon", "oganesson"}


def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip().strip("`").strip()


def _json_from(text: str):
    """Parse JSON from a reply. A Markdown code fence around the JSON is tolerated (and counted separately by
    `fenced`), because a runtime that consumes structured output strips fences anyway."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def _num_from(text: str):
    text = _strip(text).replace(",", "")
    m = re.fullmatch(r"\$?\s*(-?\d+(?:\.\d+)?)\s*%?\.?", text)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", text)
    if m:
        return int(m.group(1)) / int(m.group(2))
    return None


def parse_tool_call(text: str):
    """Accept the common open formats: <tool_call>{json}</tool_call>, bare JSON, or name(args)."""
    text = _strip(text)
    for pat in (r"<tool_call>\s*(\{.*?\})\s*</tool_call>", r"```(?:json|tool_code|tool_call)?\s*(\{.*?\})\s*```", r"^(\{.*\})$"):
        m = re.search(pat, text, flags=re.S)
        if m:
            try:
                d = json.loads(m.group(1))
            except Exception:
                continue
            if isinstance(d, dict) and "name" in d:
                args = d.get("arguments", d.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                return {"name": d["name"], "arguments": args}
    # XML-style (Qwen3.5): <function=name><parameter=key>value</parameter>...</function>
    m = re.search(r"<function=(\w+)>(.*?)</function>", text, flags=re.S)
    if m:
        args = {}
        for k, v in re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", m.group(2), flags=re.S):
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
        return {"name": m.group(1), "arguments": args}
    # Gemma-style: call:name{key:value,...} or name(key=value)
    m = re.search(r"(?:call:)?(\w+)\s*[\({](.*?)[\)}]\s*(?:<|$)", text, flags=re.S)
    if m and m.group(1) in {t["function"]["name"] for t in TOOLS}:
        args = {}
        for k, v in re.findall(r"(\w+)\s*[:=]\s*(<\|\"\|>.*?<\|\"\|>|\"[^\"]*\"|'[^']*'|[^,]+)", m.group(2)):
            v = v.strip().replace('<|"|>', "").strip("\"'")
            try:
                v = json.loads(v)
            except Exception:
                pass
            args[k] = v
        return {"name": m.group(1), "arguments": args}
    return None


def run_code(reply: str, fn: str, tests: list[str]) -> bool:
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, flags=re.S)
    src = m.group(1) if m else reply
    if f"def {fn}" not in src:
        return False
    prog = src + "\n\n" + "\n".join(tests) + "\nprint('OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        # Isolated interpreter, no site/user paths, hard time limit. Model output is untrusted code.
        r = subprocess.run([sys.executable, "-I", "-S", path], capture_output=True, text=True, timeout=10,
                           cwd=tempfile.gettempdir(), env={"PATH": "/usr/bin:/bin"})
        return r.returncode == 0 and r.stdout.strip().endswith("OK")
    except Exception:
        return False


def score(task: dict, reply: str) -> bool:
    c = task["check"]
    kind = c["type"]
    text = _strip(reply)
    if kind == "exact":
        a, b = text.rstrip("."), c["answer"]
        return a.lower() == b.lower() if c.get("fold") else a == b
    if kind == "regex":
        return re.search(c["pattern"], text) is not None
    if kind == "numeric":
        v = _num_from(text)
        return v is not None and abs(v - c["answer"]) <= c["tol"]
    if kind == "abstain":
        return text.rstrip(".").upper() == "NOT IN CONTEXT"
    if kind == "groups":
        low = text.lower()
        return all(any(k.lower() in low for k in g) for g in c["groups"])
    if kind == "code":
        return run_code(reply, c["fn"], c["tests"])
    if kind == "route":
        d = _json_from(reply)
        if not isinstance(d, list) or not all(isinstance(x, str) for x in d):
            return False
        s = set(d)
        return set(c["must"]) <= s and s <= set(c["allowed"]) and not (s & set(c["forbid"]))
    if kind == "tool":
        call = parse_tool_call(reply)
        if not call or call["name"] != c["name"] or not isinstance(call["arguments"], dict):
            return False
        args = call["arguments"]
        for k, v in c.get("args", {}).items():
            got = args.get(k)
            if isinstance(v, (int, float)):
                try:
                    if abs(float(got) - v) > 1e-6:
                        return False
                except Exception:
                    return False
            elif str(got).strip().lower() != str(v).lower():
                return False
        for k, opts in c.get("arg_fold", {}).items():
            if str(args.get(k, "")).strip().lower() not in opts:
                return False
        return True
    if kind == "no_tool":
        return parse_tool_call(reply) is None and "hello" in text.lower()
    if kind in ("json", "json_values", "json_lambda"):
        d = _json_from(reply)
        if d is None:
            return False
        types = {"str": str, "num": (int, float)}
        for k, t in c.get("required", {}).items():
            if not isinstance(d, dict) or not isinstance(d.get(k), types[t]) or isinstance(d.get(k), bool):
                return False
        for k, v in c.get("values", {}).items():
            if not isinstance(d, dict) or k not in d:
                return False
            got = d[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not isinstance(got, (int, float)) or abs(got - v) > 1e-6:
                    return False
            elif got != v:
                return False
        for k, v in c.get("contains", {}).items():
            if not isinstance(d, dict) or v.lower() not in str(d.get(k, "")).lower():
                return False
        if kind == "json_lambda":
            return JSON_LAMBDAS[c["name"]](d)
        return True
    if kind == "lambda":
        return LAMBDAS[c["name"]](text)
    raise ValueError(kind)


def _two_noble(text):
    lines = [l for l in text.splitlines() if l.strip()]
    return len(lines) == 2 and all(l.startswith("- ") and l[2:].strip().rstrip(".").lower() in NOBLE for l in lines)


LAMBDAS = {
    "no_e_sentence": lambda t: "e" not in t.lower() and "rain" in t.lower() and len(t.split()) >= 3 and t.count(".") <= 1,
    "two_noble_bullets": _two_noble,
}
JSON_LAMBDAS = {
    "elements_123": lambda d: isinstance(d, list) and len(d) == 3 and [x.get("atomic_number") for x in d] == [1, 2, 3]
    and [str(x.get("element", "")).lower() for x in d] == ["hydrogen", "helium", "lithium"],
    "curie_triple": lambda d: isinstance(d, dict) and "curie" in str(d.get("subject", "")).lower()
    and "discover" in str(d.get("relation", "")).lower() and "polonium" in str(d.get("object", "")).lower(),
    "canberra_conf": lambda d: isinstance(d, dict) and str(d.get("answer", "")).strip().lower() == "canberra"
    and isinstance(d.get("confidence"), (int, float)) and 0 <= d["confidence"] <= 1,
}

def fenced(reply: str) -> bool:
    return "```" in reply


if __name__ == "__main__":
    from collections import Counter
    print(len(TASKS), dict(Counter(t["cat"] for t in TASKS)))
