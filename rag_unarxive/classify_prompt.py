from llama_request import llama_request
from enum import StrEnum

SYSTEM_PROMPT_CLASSIFIER = """
You are a classifier who takes a user prompt and classify it into 3 different classes dependent on the question or
demand. The available classes are summarization, simplification and question-answering.
"""

class task_class(StrEnum):
    """
    Task class of a prompt. Simplification is more specific than summarization is more specific than question-answering.
    """
    SIMPLIFICATION = "simplification"
    SUMMARIZATION = "summarization" 
    QUESTION_ANSWERING = "question-answering"
    UNSPECIFIED = "unspecified"

def classify_prompt(prompt: str, num_council: int=1):
    """Classify the prompt into the categories: simplification, summarization, question-answering"""

    prompt_task = ("Examples are:\n\n" +
                   "Q: Please summarize the paper 'Attention is all you need'.'\n" +
                   "A: summarization\n\n" +
                   "Q: Please describe the paper 'Research on proton beam spot imaging based on pixelated gamma detector' in a few sentences.\n" +
                   "A: summarization\n\n" +
                   "Q: Please shorten the paper 'Tool as Embodiment for Recursive Manipulation'.\n " +
                   "A: summarization\n\n" +
                   "Q: Please break down the paper 'Some remarks on contractive and existence set' into short notes.\n"+
                   "A: summarization\n\n" +
                   "Q: Summarize the key points of the article 'The Role of AI in Modern Medicine' in one paragraph.\n"+
                   "A: summarization\n\n" +
                   "Q: Condense the novel '1984' into a brief summary of its main themes and events.\n"+
                   "A: summarization\n\n" +
                   "Q: Provide a bullet-point summary of the research paper on quantum computing advances.\n"+
                   "A: summarization\n\n" +
                   "Q: Break down the white paper on blockchain technology into key takeaways.\n"+
                   "A: summarization\n\n" +
                   "Q; Can you turn the presentation on climate change into a list of concise notes?\n"+
                   "A: summarization\n\n" +
                   "Q: Extract the main points from this financial report and format them into a summary for stakeholders.\n"+
                   "A: summarization\n\n" +
                   "Q: Summarize the contents of the article on inflation trends for a high school economics class.\n" +
                   "A: summarization\n\n" +
                   "Q: Create a short executive summary of the annual business review for the board of directors.\n"+
                   "A: summarization\n\n" +
                   "Q: Condense Chapter 5 of 'A Brief History of Time' into three sentences.\n"+
                   "A: summarization\n\n" +
                   "Q: What are the key ideas from the conclusion of the report on global trade policies?\n"+
                   "A: summarization\n\n" +
                   "Q: Summarize the similarities and differences between the U.S. Constitution and the U.K.'s unwritten constitution.\n"+
                   "A: summarization\n\n" +
                   "Q: Can you please explain the paper '' for a highschool student?\n " +
                   "A: simplification\n\n" +
                   "Q: Simplify the paper 'Composite Weak Bosons and a New Isosinglet Particle'.\n " +
                   "A: simplification\n\n" +
                   "Q: Please explain the paper 'Lifetime of quasi-particles in the nearly-free electron metal Sodium' like I am five.\n " +
                   "A: simplification\n\n" +
                   "Q: I am a undergrad and can't understand the paper 'Learning Transformer Features for Image Quality Assessment'. Please explain.\n "+
                   "A: simplification\n\n" +
                   "Q: Explain the concept of quantum entanglement in simple terms for a beginner.\n"+
                   "A: simplification\n\n" +
                   "Q: Can you simplify this legal document so that someone without a law background can understand it?\n"+
                   "A: simplification\n\n" +
                   "Q: Rewrite the research findings on CRISPR gene editing in a way that a high school student could understand.\n"+
                   "A: simplification\n\n" +
                   "Q: Break down the complex terms in this medical report for a non-medical audience.\n"+
                   "A: simplification\n\n" +
                   "Q: Simplify the technical manual for this software into easy-to-follow instructions.\n"+
                   "A: simplification\n\n" +
                   "Q: Explain the economic term 'marginal utility' in plain, everyday language.\n"+
                   "A: simplification\n\n" +
                   "Q: Can you make this philosophical argument understandable for a child?\n"+
                   "A: simplification\n\n" +
                   "Q: Simplify this engineering article for someone without a background in science.\n"+
                   "A: simplification\n\n" +
                   "Q: Rephrase this scientific journal article on climate change for the general public.\n"+
                   "A: simplification\n\n" +
                   "Q: Explain how photosynthesis works in a way that's easy to grasp, step by step.\n"+
                   "A: simplification\n\n" +
                   "Q: Simplify the process of how blockchain works into a few easy-to-follow steps.\n"+
                   "A: simplification\n\n" +
                   "Q: Can you break down the steps of filing taxes in a way that's easy for first-time filers?\n"+
                   "A: simplification\n\n" +
                   "Q: This paragraph is too complex. Can you simplify it into a shorter, clearer sentence?\n"+
                   "A: simplification\n\n" +
                   "Q: Turn this academic abstract into an easy-to-understand text.\n"+
                   "A: simplification\n\n" +
                   "Q: Rewrite this article on AI algorithms so that it’s accessible to someone with no prior knowledge of computer science.\n"+
                   "A: simplification\n\n" +
                   "Q: Explain this topic using a metaphor or analogy that’s simple to understand.\n"+
                   "A: simplification\n\n" +
                   "Q: Can you describe the Pythagorean theorem in a way that would make sense to someone who’s never taken geometry?\n"+
                   "A: simplification\n\n" +
                   "Q: Simplify this recipe into three straightforward steps for a beginner cook.\n"+
                   "A: simplification\n\n" +
                   "Which class between simplification, summarization and question-answering is the following prompt: \"{}\"?\n"
                   "\n ".format(prompt)
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_CLASSIFIER
        },
        {"role": "user", "content": prompt_task}
    ]

    this_class = task_class.UNSPECIFIED
    classes = []
    for member in range(num_council):
        chat = llama_request(messages, port=8000)
        answer = chat['generated_text'][2]['content']
        curr_class = task_class.UNSPECIFIED
        sum_idx = answer.find("summarization")
        simpl_idx = answer.find("simplification")
        qa_idx = answer.find("question-answering")

        if sum_idx<simpl_idx and sum_idx<qa_idx and sum_idx!=-1: curr_class = task_class.SUMMARIZATION
        elif simpl_idx<qa_idx and simpl_idx!=-1:                 curr_class = task_class.SIMPLIFICATION
        elif qa_idx!=-1:                                         curr_class = task_class.QUESTION_ANSWERING
        classes = classes + [curr_class]

    num_sum   = classes.count(task_class.SUMMARIZATION)
    num_simpl = classes.count(task_class.SIMPLIFICATION)
    num_qa    = classes.count(task_class.QUESTION_ANSWERING)

    if num_sum>num_simpl and num_sum>num_qa: this_class = task_class.SUMMARIZATION
    elif num_simpl>num_qa:                   this_class = task_class.SIMPLIFICATION
    else:                                    this_class = task_class.QUESTION_ANSWERING

    return this_class
