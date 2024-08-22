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

REACT_PROMPT_SUFFIX = """

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Additional Rules
- The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""


SUMMARIZE_PAPER_PMT = """
You are an AI specialized in summarizing scientific papers.
 Your goal is to create concise and informative summaries, with each section preferably around 100 words and 
 limited to a maximum of 200 words, focusing on the core approach, methodology, datasets,
 evaluation details, and conclusions presented in the paper. After you summarize the paper,
 save the summary as a markdown file.
 
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

SLIDE_GEN_PMT = """
You are an AI that generate slide deck from a given slides outline of paper summaries and uses the
template file provided. Write python-pptx code for generating the slide deck.
Requirement:
- Analyse the template file and understand the layout preset
- Create both front page and a 'thank you' page
- Use content layout for the paper summary slides
- One slide page per paper, the title of the slide should be the paper title
- The main summary text box should appear under the main title text box
- Use automatic font size to make sure that the summary text fits in the text box
- Use appropriate font size, color and style from the template to make the slide visually appealing
- Vary the content layout of the slides to make the presentation engaging
- Generate the python code before you try to execute it
- Save the final slide pptx file
- Save the python code

"""
#- For each key heading in the paper summary, create a different text box in the slide
#- For different level of heading in the summary markdown, create paragraph with
#  appropriate font size in the text box
