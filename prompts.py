IS_CITATION_RELEVANT_PMT = """
You are helping a researcher decide whether a paper is relevant to their current research topic:
Use Machine learning, NLP or GenAI technologies for automating powerpoint presentation slide generation, 
or use genAI for other layout design task.

You are given the title and abstract of a paper.
title: {title}
abstract: {abstract}

Give a score indicating the relevancy to the research topic, where:
Score 0: Not relevant
Score 1: Somewhat relevant
Score 2: Very relevant

Answer with integer score 0, 1 or 2 and your reason.
"""

RESEARCH_TOPIC = """
Use Machine learning, NLP or GenAI technologies for automating powerpoint presentation slide generation, 
or use genAI for other layout design task.
"""