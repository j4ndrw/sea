# Current pain points

1. I tried implementing a semantic router but failed spectacularly.
2. I tried to overcome the context challenge by fragmenting it, putting it a
vector store and letting the LLM pull it whenever it needs it, but that also
failed. The model never calls the vector db, just starts basing its response
off of the short-term memory, scoped per generation pass

Might've been obvious, but, unless you use a big LLM
(maybe 14B or 70B+? don't know, I don't have resources to test this) that has a big enough context size,
you can't really do anything self-evolving.

So, GPT, Claude, Gemini or other models might work fine even without the
fragmented memory + semantic router approach, but small models are simply tough to make work.
