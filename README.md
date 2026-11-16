# 530-Final-Project
# Before We Gather

A working prototype that helps community discussion facilitators
prepare for thoughtful conversations about political and civic issues.

## Current functionality

Users describe a discussion topic and optionally choose to understand
concerns or explore different perspectives.

The app produces:
- A brief grounded in retrieved ANES 2024 survey responses.
- Current web context.
- Evidence limitations.
- Generated facilitation suggestions.
- Expandable survey answers with party affiliation and predicted topics.

## Progress toward assignment requirements

### 1. Functional interface

The Streamlit interface includes a request form, preparation goals,
loading and error states, a conversation brief, expandable evidence,
and a text-download option.

The visual design uses newspaper-inspired typography, a warm background,
and restrained red and blue accents.

### 2. LLM integration and agentic workflow

A smolagents tool-calling agent using OpenAI's gpt-4.1-mini can:
1. Search survey answers using embedding similarity.
2. Search within party groups when comparing perspectives.
3. Search the web for current context.
4. Generate an evidence-backed brief and discussion suggestions.

Survey retrieval uses text-embedding-3-small with 256 dimensions.

### 3. Traditional machine learning

A trained multilabel classifier uses TF-IDF features and one-vs-rest
logistic regression to predict issue topics from survey text.

Predicted topics accompany retrieved answers, are supplied to the agent
as organizational hints, and are displayed in the evidence panel.
They do not filter or change retrieval ranking.

Held-out test performance:
- Micro precision: 0.822
- Micro recall: 0.753
- Micro F1: 0.786

## Run the prototype

Python 3.12 is used in the working Codespaces environment.

For a new environment:

    python -m venv .venv312
    ./.venv312/bin/python -m pip install -r requirements.txt

The following prepared files must be available:

    Final Project/
      responses_prepared.csv
      issue_classifier.joblib
      ml_documentation.json
      Embeddings/
        response_embeddings.npy
        embedding_response_ids.csv
        embedding_documentation.json

The code alone is not sufficient to run the app without these assets.

Set an OpenAI API key in the same Bash terminal session:

    read -rsp "OpenAI API key: " OPENAI_API_KEY
    export OPENAI_API_KEY
    echo

Start the app:

    ./.venv312/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.fileWatcherType none

In Codespaces, open forwarded port 8501.

API calls require an OpenAI API account and may incur charges.
Never commit API keys.

File watching is disabled in this launch command to avoid watcher
overflow warnings encountered in Codespaces. Restart the app after
code edits.

## Evaluation in progress

evaluate.py contains 20 ordinary, difficult, unsupported, and adversarial
requests. Outputs are manually scored on Evidence accuracy and Usefulness,
each from 0 to 2. Response time is also recorded.

Run a two-question trial:

    ./.venv312/bin/python evaluate.py --limit 2

Run all 20:

    ./.venv312/bin/python evaluate.py

Each run creates a timestamped benchmark_results folder.
The full benchmark and baseline comparison are not yet complete.

## Limitations and next steps

- Retrieved national survey examples do not represent local community
  beliefs or establish population percentages.
- Party labels support individual attribution, not party-wide conclusions.
- Classifier tags can be wrong.
- Generated claims, citations, and web summaries require review.
- Historical survey trends are not yet implemented.
- Remaining work includes evaluation, UX refinement, and final delivery.