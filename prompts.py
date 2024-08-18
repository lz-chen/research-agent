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

LLAMAPARSE_INSTRUCTION = """
This is a paper from arXiv that you need to parse. Make sure to parse it into proper markdown format.
"""

SUMMARIZE_PAPER_PMT = """
Objective:
You are an AI specialized in summarizing scientific papers.
 Your goal is to create concise and informative summaries, with each section preferably around 100 words and 
 limited to a maximum of 200 words, focusing on the core approach, methodology, datasets,
 evaluation details, and conclusions presented in the paper.
 
Instructions:
- Key Approach: Summarize the main approach or model proposed by the authors.
 Focus on the core idea behind their method, including any novel techniques, algorithms, or frameworks introduced.
- Key Components/Steps: Identify and describe the key components or steps in the model or approach.
 Break down the architecture, modules, or stages involved, and explain how each contributes to the overall method.
- Model Training/Finetuning: Explain how the authors trained or finetuned their model.
 Include details on the training process, loss functions, optimization techniques, 
 and any specific strategies used to improve the model’s performance.
- Dataset Details: Provide an overview of the datasets used in the study.
 Include information on the size, type and source. Mention whether the dataset is publicly available
 and if there are any benchmarks associated with it.
- Evaluation Methods and Metrics: Detail the evaluation process used to assess the model's performance.
 Include the methods, benchmarks, and metrics employed.
- Conclusion: Summarize the conclusions drawn by the authors. Include the significance of the findings, 
any potential applications, limitations acknowledged by the authors, and suggested future work.

Ensure that the summary is clear and concise, avoiding unnecessary jargon or overly technical language.
 Aim to be understandable to someone with a general background in the field.
 Ensure that all details are accurate and faithfully represent the content of the original paper. 
 Avoid introducing any bias or interpretation beyond what is presented by the authors. Do not add any
 information that is not explicitly stated in the paper. Stick to the content presented by the authors.
"""