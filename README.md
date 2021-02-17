# Open-Retrieval Conversational Machine Reading

OR-ShARC is an **Open-Retrieval** Conversational Machine Reading dataset focussing on answering high-level questions from texts on natural language rules. It is adapted from the [ShARC](https://sharc-data.github.io/data.html) dataset with a rewriting of all initial questions into their complete and unambiguous formats (redistributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)).

More details can be found on our [Technical report](https://arxiv.org/) (to-be-updated).

## Dataset

`id2snippet.json` contains all rule texts in OR-ShARC as our knowledge base.

Each sample in `train|dev|test` has the following attributes (Changes to the original ShARC dataset are in **bold**):

- `utterance_id`: Unique identification code for an instance in the dataset.
- `tree_id`: A tree_id specifies a unique combination of a snippet and a question. There could be several instances with the same tree_id. This is because depending on the answer that a user provide to a follow-up question, the path of the conversation or the final answer can vary.
- `source_url`: The URL of the document containing the rule snippet.
- **`snippet`**: In ShARC, it is the input support document, i.e. often a paragraph which contains some rules. **But we remove this in our dataset for our open-retrieval setting, you can refer to the `gole_snippet_id` to find the gold snippet.**
- **`gole_snippet_id`**: the gold snippet this sample should refer to in the database `id2snippet.json`.
- **`question`**: **Our rewritting** of the original incomplete and ambiguous question in ShARC.
- `scenario`: Describes the context of the question.
- `history`: The conversation history, i.e. a set of follow-up questions and their corresponding answers.
- `evidence`: A list of relevant information that the system should extract from the user's scenario. This information should not be included in the input.
- `answer`: The desired output of a prediction model.
- **`snippet_seen`**: For dev & test set only. It indicates whether this sample asks on rule texts (`snippet`) seen in training stage or not.



## Code and models will be released soon!