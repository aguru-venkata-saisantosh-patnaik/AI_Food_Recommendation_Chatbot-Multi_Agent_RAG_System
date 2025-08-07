# AI Food Recommendation Chatbot: Multi-Agent RAG System

## **Project Overview**

The **AI Food Recommendation Chatbot** is a sophisticated multi-agent system aimed at providing **personalized, context-aware food recommendations** through natural conversation. Built upon a Retrieval-Augmented Generation (RAG) paradigm, the system leverages advanced user modeling, conversational memory, semantic retrieval, and explainable reranking to optimize food suggestions for individual users. Its modular agents, robust data engineering, and user clustering ensure relevance, scalability, and adaptability for varying user behaviors and food contexts.

## **Workflow Overview**
<img width="3840" height="2160" alt="Intent Classification" src="https://github.com/user-attachments/assets/9415d086-b89e-461c-8293-a528999797c6" />

### **Workflow Steps**

1. **User Input:**  
   - Users engage the chatbot with food-related queries or preferences (e.g., “Show me spicy veg biryani under 300”).

2. **Conversational Agent:**  
   - **Intent Classification:** Determines the itent of the user like greeting, goodbye, preference updation, request for recommendation, specifying preference etc.
   - **Slot Extraction:** Extracts structured information from the conversation (dietary, cuisine, dish, price, etc.).
   - **Memory Updation:** Updates session memory to retain preferences and context.

3. **Sufficiency Check:**  
   - Decides if enough info is available to generate recommendations.
   - If not, the agent asks targeted follow-up questions.

4. **Retrieval Agent:**  
   - **Query Enhancer:** Transforms structured slots into robust search queries and filters.
   - **Retrieval from Shards:** Performs semantic searches over sharded databases of food items.

5. **Reranking Agent:**  
   - **Condition Generation & Evaluation:** Produces context-sensitive, explainable ranking rules based on the user’s food journey and menu diversity.
   - **Evaluator & Top 10 Identifier:** Scores and explains top 10 results, ensuring the best fit and diversity.

6. **Output:**  
   - Presents recommendations or further clarifying questions to the user, looping until satisfaction.

## **Folder Description**
- **Python Files:** This folder contains all the files related to running the program. Each part of the code is made in a modular way to make future changes easy.
- **User_clustering_file:** This folder contains all the files related to the user clustering module (K++ means trained algorithm).
- **Application:** This folder contains all the files related to the front-end made using gradio and its connection to the orchestrator.
- **Data Cleaning and Feature Engineering:** This folder contains all the files related to the preprocessing of the datasets and the feature engineering.
- **Embedding and Shards Creation:** This folder contains all the files related to embedding of the data and making of the shards. The link to the shards made is given at the end.
- **Demo_final_compressed.mp4**: This is the demo video which contains the full run of the application showing 2 scenarios of various inputs.
- **Case_presentation.pdf**: This is the presentation showcasing the importance, usability and findings of this project.
- **Report.pdf**: This is the detailed report explaining the various technical aspects of this projects.

## **Solution Architecture**

### 1. **Conversational Agent**
- **File:** `conversation_agent.py`
  - **Intent Classification:** Uses LLMs and fallback pattern recognition (`intent_classifier.py`) to detect user goals.
  - **Slot Extraction:** Employs prompt-based extraction plus robust rules (`slot_extract.py`, `query_enhancer.py`).
  - **Memory Management:** Maintains stateful conversation context (`memory.py`).
  - **Output:** A set of structured preferences and clear user intent, enabling smooth pipeline orchestration.

### 2. **Recommendation Sufficiency Check**
- **Embedded Logic:** In `conversation_agent.py`, checks for essential details (like dietary and price).
- **Diagram Mapping:** The central diamond; if information is incomplete, prompts the user; otherwise, triggers retrieval and reranking.

### 3. **Retrieval Agent**
- **Files:** `shards_retrieval.py`, (functions also exposed via `orchestrator.py`)
  - **Query Refinement:** Structures and optimizes semantic queries using `query_enhancer.py`.
  - **Sharded Retrieval:** Searches distributed ChromaDB shards via dense embeddings (`embeddings.py`), ensuring speed and scalability.

### 4. **Re-ranking Agent**
- **Files:** `rerank.py`, `rerank_prompts.py`
  - **Contextual Reranking:** Analyzes user journey, applies dynamic, explainable ranking/validation for top recommendations.
  - **Explainability:** Outputs condition reasoning and QA, making the selection process transparent and defensible.

### 5. **Embeddings & Shard Creation**
- **Files:** `embeddings.py`, `shards_creation.ipynb`
  - **Embeddings:** Utilizes sentence-transformers to vectorize each food/menu item for semantic similarity.
  - **Sharding:** Splits large databases into efficient “shards” for fast, parallel retrieval.

### 6. **User Clustering & Data Pre-processing**
- **Files:** `user_clustering_agent.py`, `User_Clustering_agent.ipynb`, `derived_feature_engineering.ipynb`, `2-cuisines.ipynb`, `zomato_restaurant_data_cleaning.ipynb`
  - **Clustering:** Segments users by behavior (demographics, purchases, cuisine affinity) using KMeans, making suggestions more relevant.
  - **Preprocessing:** Cleans, normalizes, engineers features (e.g., veg ratio, price tiers), supporting robust recommendation logic and clustering.

## **Deep Dive: What Each File Does**

| File/Notebook                       | Role & Description                                                    |
| -------------------------------------|-----------------------------------------------------------------------|
| `conversation_agent.py`              | Controls user–system dialogue, slots/intents, memory, agent pipeline  |
| `intent_classifier.py`               | LLM/fallback-based intent classification                              |
| `slot_extract.py`                    | Extracts structured input slots from user text                        |
| `memory.py`                          | Tracks all slots, conversation turns, dialogue state                  |
| `response_generator.py`              | Crafts user-facing responses and active follow-ups                    |
| `query_enhancer.py`                  | Refines and builds advanced queries and filters for retrieval         |
| `shards_retrieval.py`                | Executes distributed, parallel semantic search via ChromaDB shards     |
| `embeddings.py`                      | Handles embedding model setup and vector generation                   |
| `shards_creation.ipynb`                     | Stepwise guide to generate embeddings and shard databases             |
| `rerank.py`, `rerank_prompts.py`     | Contextual re-ranking and reasoning/validation logic                  |
| `User_Clustering_agent.ipynb`        | Clustering model training/testing pipeline and User persona recognition for enhanced recommendations                           |
| `derived_feature_engineering.ipynb`  | Data augmentation for features (ratios, groupings, etc.)              |
| `2-cuisines.ipynb`                   | Cuisine feature merging and exploration                               |
| `zomato_restaurant_data_cleaning.ipynb`| Cleans, standardizes and joins all food/restaurant data               |
| `orchestrator.py`                    | Ties all agents together; runs config, workflow, and agent init       |
| `utils.py`                           | Core enums, helper functions, configuration constants                 |
| `app.py`                           | Contains the frontend made with gradio and connects to the orchestrator                 |

## **How the Components Work Together**

- Conversations flow through the **Conversational Agent**, which extracts intent, slots, and maintains memory for multi-turn dialogues.
- Once enough preferences are gathered, the **Retrieval Agent** turns these into optimized, context-rich search queries.
- These queries search a semantic **vector DB** distributed in **shards** (for efficiency)—returning a candidate set of food items.
- The **Re-ranking Agent** processes retrieved items, using the user’s session context, journey, and explainable conditions to select the most fitting top 10.
- **User Clustering** and **engineered data features** guide personalization ensuring user traits, habits, and history are considered.

## **Major Data & Model Preparation Steps**

- **Data Sources and cleaning:**
   - Food Recommendation CSV (schemersays): "https://www.kaggle.com/datasets/schemersays/food-recommendation-system?select=1662574418893344.csv"
   - Zomato Restaurants Dataset (bharathdevanaboina): "https://www.kaggle.com/datasets/bharathdevanaboina/zomato-restaurants-dataset/data"
   - Zomato Database (anas123siddiqui): "https://www.kaggle.com/datasets/anas123siddiqui/zomato-database?select=restaurant.csv"
  These Datasets are further cleaned, deduplicated, and joined for quality and reliability.
- **Feature Engineering:**  
  Enriched with behavioral, pricing, cuisine, and sensitivity features for deep personalization.
- **Embeddings & Sharding:**  
  All items are vectorized and stored in scalable shards. The link to the shards made during the project is : "https://drive.google.com/drive/folders/1yYOu3G_TZ9srSL8hK5-hdkgP7m9wUXic?usp=sharing"
- **User Clustering:**  
  K++ Means trained clusters assign users to personas that guide ranking and filtering logic.


