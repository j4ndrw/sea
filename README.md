# Current pain points

1. Context size is overflown easily. I'll have to implement a step before every generation where I inject the knowledge base into the conversation (sensibly, via semantic search)
2. Web search tool is not implemented yet. The model seems to want to reach for it every time it gets the chance, despite the system prompt telling it not to.
In hindsight this might be a good thing - forces the model to always look for answers instead of making them up.
3. The model tends to want to reach for the knowledge base query tool often. This is currently unimplemented.
4. The model's strategies to creating agents and tools are questionable, often creating a single tool and multiple agents, rather than an agent with a collection of tools. This might not be that bad, but still need to experiment.
