# Worked example: decomposing a multi-hop question

Question: "Who directed the film that won the Academy Award for Best Picture in 1994?"

Turn 1 — ask the first sub-question as a plain string:

```repl
best_picture = llm_query("Which film won the Academy Award for Best Picture in 1994?")
print(best_picture)
```

Turn 2 — use the first answer to ask the next sub-question, then submit:

```repl
director = llm_query("Who directed the film Forrest Gump?")
answer["content"] = director
answer["ready"] = True
print(director)
```

Notice: each `llm_query(...)` receives a plain-string question, and the final answer is aggregated
from the sub-answers rather than produced directly.
