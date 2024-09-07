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

summary2outline_requriements = """

- Use the paper title as the slide title
- Use the summary in the markdown file as the slide content, convert the markdown headings to
 bullet points by prepending each heading text with a bullet (* or -). You can choose to maintain
 the hierarchy by using indentation.
- Rephrase the content to make it more concise, and straight to the point
- Each bullet point should be less than 15 words
- A paragraph of text in the slide should be less than 25 words.
"""

SUMMARY2OUTLINE_PMT = (
    """
You are an AI specialized in generating PowerPoint slide outlines based on the content provided.
You will receive a markdown string that contains the summary of papers and 
you will generate a slide outlines for each paper.
Requirements:"""
    + summary2outline_requriements
    + """

Here is the markdown content: {summary} 
"""
)

MODIFY_SUMMARY2OUTLINE_PMT = (
    """
You are an AI that modifies the slide outlines generated according to given user feedback.
The original summary is '''{summary_txt}'''.
Previously generated outline is '''{outline_txt}'''.
The feedback provided is: '''{feedback}'''.
Please modify the outline based on the feedback and provide the updated outline, respecting
 the original requirements:"""
    + summary2outline_requriements
)

AUGMENT_LAYOUT_PMT = """
You are an AI that selects slide layout from a template for the slide text given.
You will receive a page content with title and main text.
Your task is to select the appropriate layout and information such as index of the placeholder for the page 
 based on what type of the content it is (e.g. is it topic overview/agenda, 
 or actual content, or thank you message).
For content slides, make sure to 
 - choose a layout that has content placeholder (also referred to as 'Plassholder for innhold') 
 after the title placeholder
 - choose the content placeholder that is large enough for the text content
 
The following layout are available: {available_layout_names} with their detailed information:
{available_layouts}

Here is the slide content:
{slide_content}
"""

# General rules of selecting layout:
# - if the slide text is list of topics or agenda, select one of the 'Agenda' layout
# - if the slide text contains text and some paragraph, select one of the 'Content' layout
# - if the slide text is very short, depending on the text, select either 'Frontpage' or 'Thank you' layout
# - Make sure to choose a layout that has main text box placeholder after the title placeholder (this can be judged
# by the order they appear in the list of placeholders)
# - Use appropriate color and style from the template to make the slide visually appealing

SLIDE_GEN_PMT = """
You are an AI that generate slide deck from a given slide outlines and uses the
 template file provided. Write python-pptx code for generating the slide deck by loop over the slide 
 outlines provided.
You will be provided with a json file `{json_file_path}` that contains a list of slide outlines
 and layout to use from the template.
The template file is located at `{template_fpath}`.
If you can't find those files at remote location, you need to upload them.
Respond user with the python code that generates the slide deck.

Requirement:
- If there is no front page or 'thank you' page, create them by using the related layout template in the
 layout information provided, DO NOT assume the index of the layout for them
- If the placeholder chosen has text auto_size set to TEXT_TO_FIT_SHAPE, make sure to set the
 text to fit the shape (use MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE) and DO NOT set a font size
- One slide page per outline item that you are given, fill in all the title and content you are given
- Use layout and text box index according to what is given in each of the slide content item 
- Vary the content layout of the slides to make the presentation engaging
- Generate the python code before you try to execute it
- Save the final slide pptx file with name `{final_slide_fname}`

"""
# - For each key heading in the paper summary, create a different text box in the slide
# - For different level of heading in the summary markdown, create paragraph with
#  appropriate font size in the text box

SLIDE_VALIDATION_PMT = """
You are an AI that validates the slide deck generated according to following rules:
- The slide need to have a front page 
- The slide need to have a final page (e.g. a 'thank you' or 'questions' page)
- The slide texts are clearly readable, not cut off, not overflowing the textbox
 and not overlapping with other elements

If any of the above rules are violated, you need to provide the index of the slide that violates the rule,
 as well as suggestion on how to fix it. Note: missing key aspect can be due to the font size being too large
 and the text is not visible in the slide, make sure to suggest checking the original slide content texts to
 see if they exist, and reducing the font size as a solution.

"""

SLIDE_MODIFICATION_PMT = """
You are an AI that modifies the slide deck according to given feedback using python-pptx library.
The original slide deck can be found here {pptx_path}. 
The feedback provided is as follows: {feedback}.
Save the modified slide deck as {modified_pptx_path}.

"""
