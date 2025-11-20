<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Goklyn AI‑ML Internship – AI/ML Tasks

This repository contains implementations and documentation for the three AI‑ML intern assessment tasks: a tabular classification model, a simple AI agent, and a mini‑project proposal on SIEM‑focused AI agents.[^1][^2][^3][^4]

***

## Task 1 – Heart Disease Classification (`classificationmodel.ipynb`)

### Objective

This notebook implements the “ML Model Building” task by training a supervised classifier on a public heart‑disease dataset using scikit‑learn.[^2][^1]
The goal is to predict the **Heart Disease Status** label from a mix of clinical and lifestyle features and to demonstrate an end‑to‑end, reproducible ML workflow.[^2]

### Dataset and features

The model expects a CSV file named `heartdisease.csv` containing one target column `Heart Disease Status` with values such as “Yes/No” and several numerical and categorical predictors.[^2]
Typical numerical fields include information like patient age, blood pressure, cholesterol level, and BMI, while categorical fields capture attributes such as gender, smoking status, and exercise habits.[^2]

### Code structure and pipeline

The notebook uses a `TrainingConfig` dataclass to centralize configuration such as the dataset path, target column name, train–test split size, and random seed.[^2]
Data loading, feature–target splitting, type detection, pipeline construction, and evaluation are each encapsulated in dedicated functions (`load_data`, `split_features_and_target`, `get_feature_types`, `build_pipeline`, `evaluate_model`) to keep the workflow modular and easy to extend.[^2]

### Preprocessing and model

Numeric features are processed with a `SimpleImputer` using the median strategy followed by `StandardScaler` to normalize the scale of continuous variables.[^2]
Categorical features are imputed using the most frequent category and one‑hot encoded with `OneHotEncoder(handle_unknown="ignore")`, then both branches are combined with a `ColumnTransformer`.[^2]
On top of this preprocessing block, the notebook trains a `RandomForestClassifier` (200 trees, fixed `random_state`, and `n_jobs=-1` for parallelism) wrapped into a single `Pipeline` object.[^2]

### Training and evaluation

The data is split into train and test sets with `train_test_split`, using the `stratify` argument on the target to preserve class balance in both splits.[^2]
After fitting the pipeline on the training data, the notebook evaluates the model on the test set using overall accuracy and a detailed `classification_report` to inspect per‑class precision, recall, and F1 scores.[^2]
The sample run shows good accuracy dominated by the majority “No” class, while the minority “Yes” class suffers from low recall, highlighting class imbalance and motivating potential improvements such as rebalancing or tuning model hyperparameters.[^2]

### How to run

1. Place `heartdisease.csv` in the same directory as the notebook or update `TrainingConfig.datapath` to point to the correct file.[^2]
2. Install Python dependencies (`pandas`, `scikit-learn`) in your environment, then open and run the notebook cells in order.[^2]
3. At the end of execution the notebook prints test accuracy and the classification report, which can be used for comparison when experimenting with other models or preprocessing choices.[^2]

***

## Task 2 – Simple AI Agent (`ai-agent-implementation.ipynb`)

### Objective

This notebook addresses the “AI Agent Task” by building a minimal conversational agent that analyzes user input and generates context‑aware responses.[^3][^1]
The agent focuses on two lightweight NLP tasks—intent detection and sentiment estimation—before crafting a concise reply, meeting the requirement to “analyze user input and generate responses.”[^1][^3]

### LLM‑based agent (OpenAI API)

One section of the notebook uses the OpenAI Python SDK and a chat‑capable model (for example a small GPT‑4 class model) to implement an analysis‑and‑reply function.[^3]
The code reads an API key from Kaggle Secrets, sends the user’s message and a system prompt to the model, and expects a strictly JSON‑formatted output with three fields: `intent`, `sentiment`, and `reply`.[^3]
A helper function parses the JSON response, applies fallbacks if parsing fails, and returns a clean Python dictionary that can be consumed by downstream logic or a console chat loop.[^3]

### Rule‑based + LangChain agent

To provide a dependency‑light alternative that does not require external API calls, the notebook also defines a fully rule‑based agent composed of three core functions: `detect_intent`, `analyze_sentiment`, and `craft_reply`.[^3]
`detect_intent` uses simple keyword and phrase rules to assign labels such as greeting, gratitude, information request, farewell, or general conversation, while `analyze_sentiment` counts positive and negative terms to return a coarse sentiment (positive, neutral, or negative).[^3]
`craft_reply` then combines the inferred intent and sentiment to generate a short, deterministic answer—for example, friendly greetings for salutations, explanations of agent capabilities for information requests, or more supportive wording when negative sentiment is detected.[^3]

### LangChain pipeline and console loop

The rule‑based analysis is wrapped into LangChain `RunnableLambda` components to form a simple chain: raw user text → analysis dict → enriched dict with reply.[^3]
A small `chatloop` function implements an interactive REPL where the user types messages, the chain is invoked, and the notebook prints the detected intent, detected sentiment, and the generated reply, exiting when the user types `exit`, `quit`, or `q`.[^3]
This design cleanly demonstrates the core AI‑agent pattern requested in the assignment: a pipeline that interprets user input, attaches semantic labels, and produces structured, human‑readable responses.[^1][^3]

### How to run

1. For the OpenAI‑based variant, configure the `OPENAI_API_KEY` secret in the Kaggle environment (or adapt the key loading section to your own environment) and install the `openai` Python package.[^3]
2. For the rule‑based LangChain variant, install the listed dependencies (such as `langchain` and `langchain-huggingface` where used) and run the notebook cells defining the helper functions and the `chatloop`.[^3]
3. Execute the final cell to start the console chat, type a few natural language messages (greetings, questions, or farewells), and observe how the agent annotates intent and sentiment before replying.[^3]

***

## Task 3 – Mini‑Project Proposal: SIEM Co‑Pilot Agent

### Problem context

Modern SIEM deployments already ingest and correlate large volumes of security logs, yet SOC teams still struggle with noisy alerts, static correlation rules, and slow manual investigations.[^4]
Many environments rely on a standard “20‑rule” baseline pack that covers critical scenarios—such as brute‑force logins, privilege misuse, lateral movement, malware activity, and data exfiltration—but those rules are seldom continuously tuned for changing environments.[^4]
The proposal focuses on augmenting these existing correlation rules with AI agents rather than replacing them, with the aim of reducing noise and accelerating response for security analysts.[^4]

### Proposed SIEM Co‑Pilot architecture

The mini‑project introduces a **SIEM Co‑Pilot** composed of three cooperating agents that sit on top of an existing SIEM and SOAR stack and operate on alerts generated by the baseline rules.[^4]
A Detection Triage Agent monitors incoming alerts, groups similar events, and assigns a simple risk score based on factors like asset criticality, user role, and correlated event counts to prioritize analyst attention.[^4]
A Context Enrichment Agent automatically queries external systems—asset inventory, vulnerability scanners, EDR tools, and threat‑intelligence feeds—to gather additional evidence for high‑risk alerts, emulating an analyst’s investigation steps but in a much shorter time.[^4]
Finally, a Response Orchestration Agent uses predefined playbooks to suggest or trigger actions such as isolating endpoints, disabling compromised accounts, blocking malicious IPs, or creating incident tickets, with different behavior for low‑confidence versus high‑confidence situations.[^4]

### Applying the 20 SIEM rules

In this design the twenty baseline SIEM rules form the signal layer, and the agents make these signals more actionable instead of rewriting correlation logic.[^4]
For authentication‑related detections (for example password spraying or impossible travel), the agents correlate identity, VPN, IAM, and endpoint logs to distinguish benign user mistakes from true attacks and gradually refine thresholds.[^4]
For admin and privilege‑escalation events, they cross‑reference maintenance windows, approved change tickets, and vulnerability data before determining whether to raise a high‑priority incident.[^4]
Malware‑oriented rules are enriched with endpoint process trees and sandbox verdicts so that the risk score increases only when there are signs of execution or lateral movement, reducing the volume of noisy detections.[^4]
For data‑exfiltration and network‑anomaly scenarios, the agents incorporate business context and normal traffic baselines for a given user or host, aligning decisions with behavior‑driven SIEM best practices.[^4]
Over time, analyst feedback on which alerts were true or false positives is fed back into the agents so they can learn which of the 20 rules are most valuable in that specific environment.[^4]

### Scope, environment, and evaluation

The project can initially be implemented in a lab or trial SIEM setup using a limited but realistic dataset (for example VPN logs, Windows Active Directory events, and EDR telemetry) plus a lightweight SOAR platform or scripting layer for automated actions.[^4]
Success metrics include comparing pre‑ and post‑deployment values for alert volume from the 20 rules, proportion of alerts automatically closed as benign, triage time for high‑severity alerts, and the number of correctly automated containment actions.[^4]
By the end of the mini‑project, the outcome is a working prototype that demonstrates how a small set of AI agents can sit on top of a traditional rule‑based SIEM, make the canonical rule pack more effective, and push the SOC toward a more autonomous operating model.[^4]

<div align="center">⁂</div>

[^1]: Internship_Assignment-Goklyn.pdf

[^2]: classificationmodel.ipynb

[^3]: ai-agent-implementation.ipynb

[^4]: Mini-project-proposal.pdf

