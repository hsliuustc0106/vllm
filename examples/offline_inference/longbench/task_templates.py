TASK_TEMPLATES = {

    "narrativeqa": {
        "template": ("You are given a story, which can be either a novel or a movie script, and a question. "
                    "Answer the question as concisely as you can, using a single phrase if possible. "
                    "Do not provide any explanation.\n\nStory: {context}\n\n"
                    "Now, answer the question based on the story as concisely as you can, "
                    "using a single phrase if possible. Do not provide any explanation.\n\n"
                    "Question: {input}\n\n"),
        "answer_prefix": "Answer:"
    },
    
    "qasper": {
        "template": ("You are given a scientific article and a question. Answer the question as concisely as you can, "
                    "using a single phrase or sentence if possible. If the question cannot be answered based on "
                    "the information in the article, write \"unanswerable\". If the question is a yes/no question, "
                    "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
                    "Article: {context}\n\nAnswer the question based on the above article as concisely as you can, "
                    "using a single phrase or sentence if possible. If the question cannot be answered based on "
                    "the information in the article, write \"unanswerable\". If the question is a yes/no question, "
                    "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
                    "Question: {input}\n\n"),
        "answer_prefix": "Answer:"
    },

    "multifieldqa_en": {
        "template": ("Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question "
                     "based on the above text, only give me the answer and do not output any other words.\n\n"
                     "Question: {input}\n"),
        "answer_prefix": "Answer:"
    },

    "hotpotqa": {
        "template": ("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                     "The following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                     "Only give me the answer and do not output any other words.\n\n"
                     "Question: {input}\n"),
        "answer_prefix": "Answer:"
    },

    "2wikimqa": {
        "template": ("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                     "The following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                     "Only give me the answer and do not output any other words.\n\n"
                     "Question: {input}\n"),
        "answer_prefix": "Answer:"
    },

    "musique": {
        "template": ("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                     "The following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
                     "Only give me the answer and do not output any other words.\n\n"
                     "Question: {input}\n"),
        "answer_prefix": "Answer:"
    },
    
    "gov_report": {
        "template": ("You are given a report by a government agency. Write a one-page summary of the report.\n\n"
                   "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\n"),
        "answer_prefix": "Summary:"
    },

    "qmsum": {
        "template": ("You are given a meeting transcript and a query containing a question or instruction. "
                     "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
                     "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
                     "Query: {input}\n"),
        "answer_prefix": "Answer:"
    },

    "multi_news": {
        "template": ("You are given several news passages. Write a one-page summary of all news. \n\n"
                     "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\n"),
        "answer_prefix": "Summary:"
    },

    "trec": {
        "template": ("Please determine the type of the question below. "
                     "Here are some examples of questions.\n\n{context}\n\n{input}"),
        "answer_prefix": ""
    },

    "triviaqa": {
        "template": ("Answer the question based on the given passage. Only give me the answer and do not output any other words. "
                     "The following are some examples.\n\n{context}\n\n{input}"),
        "answer_prefix": ""
    },

    "samsum": {
        "template": ("Summarize the dialogue into a few short sentences. "
                     "The following are some examples.\n\n{context}\n\n{input}"),
        "answer_prefix": ""
    },

    "passage_count": {
        "template": ("There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
                     "Please carefully read these paragraphs and determine how many unique paragraphs there are "
                     "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
                     "{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. "
                     "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"),
        "answer_prefix": "The final answer is: "
    },

    "passage_retrieval_en": {
        "template": ("Here are 30 paragraphs from Wikipedia, along with an abstract. "
                     "Please determine which paragraph the abstract is from.\n\n"
                     "{context}\n\nThe following is an abstract.\n\n{input}\n\n"
                     "Please enter the number of the paragraph that the abstract is from. "
                     "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"),
        "answer_prefix": "The answer is: "
    },

    "lcc": {
        "template": ("Please complete the code given below. \n{context}"),
        "answer_prefix": "Next line of code:\n"
    },

    "repobench-p": {
        "template": ("Please complete the code given below. \n{context}{input}"),
        "answer_prefix": "Next line of code:\n"
    },

}