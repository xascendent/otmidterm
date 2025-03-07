# AIE5_Midterm

Links for turn in:
https://www.loom.com/share/8f4efe7f4fd647fc8bb3ed0353f794ac

https://huggingface.co/shivXy/ot-midterm-v0

https://huggingface.co/spaces/shivXy/otmidterm

https://github.com/xascendent/otmidterm

Location of docs that I will use for my vector store:
https://journals.lww.com/iopt/fulltext/2018/50040/effect_of_hand_splinting_versus_stretching.5.aspx

Homework location:
https://www.notion.so/Session-11-Midterm-189cd547af3d800d8407f74826756268#189cd547af3d81c2b2fbc0573761ab21

 What specific therapeutic activities and exercises have been shown to be most effective in resolving symptoms and treating chronic tennis elbow for adults between the ages of 30-50 years old according to peer reviewed journal articles published in the last 5 years? 

break out like this:
Research indicates several therapeutic activities and exercises effective for treating chronic tennis elbow (lateral epicondylitis) in adults aged 30-50. Here are the most commonly supported interventions based on evidence from peer-reviewed studies published in the last five years:

1. **Eccentric Exercises**: Eccentric strengthening exercises targeting the wrist extensors have been shown to effectively reduce pain and improve function.

2. **Isometric Exercises**: Isometric exercises can help build strength in the forearm muscles without increasing pain, making them suitable for acute phases of recovery.

3. **Stretching**: Gentle stretching of the wrist extensors improves flexibility and alleviates tension, contributing to pain relief.

4. **Manual Therapy**: Soft tissue mobilization and joint mobilization techniques may enhance outcomes, especially when combined with exercise.

5. **Ultrasound Therapy**: While research on therapeutic ultrasound shows mixed results, some studies support its use alongside exercise therapy for symptom relief.

6. **Taping and Bracing**: Kinesiology tape or forearm braces can provide support, reducing strain during activities and allowing for better engagement in therapeutic exercises.

7. **Functional Activities**: Gradual return to specific functional tasks, customized to the individual's daily activities, can improve physical readiness and psychological well-being.

Combining these interventions in a comprehensive treatment plan, tailored to the individual's specific condition and needs, is crucial for effective management of chronic tennis elbow. Always consider consulting the latest literature or a healthcare professional for updated strategies.



powershell commands:
- create a git repo in github have the creator add the python git ignore file
- uv init src_midterm
- cd src_midterm
- del hello.py
- uv add uuid7 nest_asyncio langchain_core langgraph langchain-text-splitters langchain_community langchain-qdrant langchain_openai    
- Open vs code . code no longer works since we installed the other IDE (Cursor)
- Let vsCode control the git

---

## MIDTERM CORRECTIONS:
Task 2: Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?
`Answer`: The agentic part comes from the way the system is designed to take actions based on intermediate reasoning steps rather than just retrieving documents and passing them to the model.  You will see this in the nodes
and the steps they take from looking at user input, document results, etc.

Task 3: [Optional] Will you need specific data for any other part of your application?   If so, explain.
`Answer`: I pull specific data from Journal of Hand Therapy

Task 4: "Consider making your prototype generic. It was not able to give answers to other questions, and it always gave multiple lines for a list of predefined sections.

e.g question - How do hand splinting and stretching exercises compare in their effectiveness at reducing spasticity and improving hand function in poststroke patients?"

`Answer`: I have removed all the RAGAS code since this task is better suited in other areas.  Thus simplifiy the code.  I also made the code to where you can ask other types of questions such as tell me a joke 
and it will not try to format everything.  It will only try to format if the LLM thinks that the question is an OT question.  You can find the updated space here: https://huggingface.co/spaces/shivXy/midtermfixes


Task 5: Creating a Golden Test Data Set
`Answer`: I have created a new py file that does all the evals for RAGAS and LANGSMITH.  I evaluated multiple LLMs with different embeddings including my own embedding which can be found here: https://huggingface.co/shivXy/ot-midterm-v0  the new code to do the evals can be found here: https://github.com/xascendent/otmidterm/tree/main/Evals  you will find the langsmith screenshot.  but the RAGAS metrics 
that are being looked for are saved out to a file for each run / appened.  here is a break out of all the models and embeddings used:
'Model Used': 'gpt-3.5-turbo' Embeddings Model Used': 'text-embedding-3-small' 'RESULTS':{'faithfulness': 0.5753, 'factual_correctness': 0.2845, 'answer_relevancy': 0.3537, 'context_entity_recall': 0.4855}
'Model Used': 'gpt-4o' Embeddings Model Used': 'text-embedding-3-small' 'RESULTS':{'faithfulness': 0.8829, 'factual_correctness': 0.4182, 'answer_relevancy': 0.9844, 'context_entity_recall': 0.0790}
'Model Used': 'gpt-3.5-turbo' Embeddings Model Used': 'text-embedding-3-large' 'RESULTS':{'faithfulness': 0.8409, 'factual_correctness': 0.4327, 'answer_relevancy': 0.5341, 'context_entity_recall': 0.2617}
'Model Used': 'gpt-4o' Embeddings Model Used': 'text-embedding-3-large' 'RESULTS':{'faithfulness': 0.8605, 'factual_correctness': 0.5255, 'answer_relevancy': 0.8948, 'context_entity_recall': 0.0769}
'Model Used': 'gpt-4o' 'Embeddings Model Used': 'huggingface/shivXy/ot-midterm-v0' 'RESULTS':{'faithfulness': 0.8690, 'factual_correctness': 0.5667, 'answer_relevancy': 0.7774, 'context_entity_recall': 0.2458}
'Model Used': 'gpt-3.5-turbo' 'Embeddings Model Used': 'huggingface/shivXy/ot-midterm-v0' 'RESULTS':{'faithfulness': 0.8333, 'factual_correctness': 0.4017, 'answer_relevancy': 0.3178, 'context_entity_recall': 0.4026}
'Model Used': 'gpt-4o' 'Embeddings Model Used': 'huggingface/shivXy/ot-midterm-v0' 'RESULTS':{'faithfulness': 0.9481, 'factual_correctness': 0.6867, 'answer_relevancy': 0.7792, 'context_entity_recall': 0.1602}

What conclusions can you draw about the performance and effectiveness of your pipeline with this information?
`Answer`: As you can see from this chart and the langsmith i went with the custom hugging face for my qdrant embeddings.  I'm still using qdrant cloud for my datastore.


Task 6: Fine-Tuning Open-Source Embeddings

`Answer`: https://huggingface.co/shivXy/ot-midterm-v0  You can see the metrics above and I used this for my final embedding with gtp-4o

Task 6: Assessing Performance
Question: How does the performance compare to your original RAG application? Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements. Provide results in a table.
`Answer`: The performance due to all the calls is slow and costly.  But the data accuracy is much higher based on my tunings and deployment.  You can see the results from task 5.

Quetion: Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?
`Answer`: The second half of the course I would love to see if I could get this out of the OPENAI ecosystem.  The bills are pretty high with all of the debugging/testing cycles.  I'm hopeful that will perform at around the
same levels or close to.  At least I will ahve the golden py file that I can swap in and out models for testing.


Task FINAL: 
New Loom video I will just go over the changes that was made to the application.  The old loom video overall work flow is still relevant.  New loom video link:
Public Github Repo : https://github.com/xascendent/otmidterm
Public Github Repo / RAGAS / EVALS : https://github.com/xascendent/otmidterm/tree/main/Evals  NonBranchRevs are multiple tries of the project until I was able to get golden4.py 
Public Github REPO New App : https://github.com/xascendent/otmidterm/blob/main/app.py
Public Github REPO Custom model: https://github.com/xascendent/otmidterm/tree/main/fine%20tuning%20project
Public Github REPO First Project attempt: https://github.com/xascendent/otmidterm/tree/main/Real%20project  (this is my first attempt at the project but HF doesn't like multiple py files etc. had to start over)

A written document addressing each deliverable and answering each question: This file has all corrections and all answered questions from below.  I figured that you wanted me to answer you the areas that needed corrections 
thus I will leave the below task/questions/answers the same and just address what was asked for.

Hugging Face: https://huggingface.co/shivXy/ot-midterm-v0  embedding model 
Hugging Face: https://huggingface.co/spaces/shivXy/midtermfixes  new app 

## Task 1: 

**Problem Statement:**  
Occupational therapists often need quick access to reliable, up-to-date information on splinting techniques and tools while working with patients, but the time-consuming process of manually searching for articles and images can disrupt patient care.

**Why this is a Problem:**  
Occupational therapists, like my wife, are often pressed for time during patient sessions and need immediate access to accurate and relevant information. Manually searching through journal articles and Google for splinting techniques or images is inefficient and detracts from valuable patient interaction. A chatbot capable of quickly retrieving and displaying OT splinting information, including images or at least direct links to source articles, would streamline this process. This solution would enhance clinical efficiency, support evidence-based practice, and ultimately improve patient outcomes by allowing the therapist to focus more on hands-on care rather than administrative tasks.

## Task 2:
Proposed Solution:

This solution will provide an intuitive chatbot interface that allows the occupational therapist to quickly ask questions about splinting techniques, retrieve relevant journal articles, and view images directly within the chat (if possible). The chatbot will streamline the search process by automatically querying reliable sources and returning the most pertinent results, ensuring that the therapist can focus on patient care.

Tooling Choices:

a. `LLM` – OpenAI's models will be used initially for its state-of-the-art performance in natural language understanding and generation. While the assignment may require fine-tuning a model later, GPT-4o-mini provides a solid proof of concept due to its reliability and ease of integration.

b. `Embedding Model` – OpenAI's embedding model will be used to generate embeddings for text and images, ensuring semantic search capabilities across both modalities.

c. `Orchestration` – LangGraph will be utilized for its graph-based orchestration, allowing for flexible and maintainable workflows when managing multiple retrieval and processing steps.

d. `Vector Database` – Qdrant will serve as the vector database because it supports multimodal storage, making it ideal for storing both image embeddings and PDF embeddings separately while allowing for efficient combination during retrieval.

e. `Monitoring` – LangSmith will be used for monitoring, as it provides detailed tracking, debugging, and visualization of LLM application workflows, ensuring that the system runs smoothly and any issues can be quickly identified and resolved.

f. `Evaluation` – RAGAS will be employed for evaluation, particularly for assessing the quality of responses in a RAG (Retrieval-Augmented Generation) system. This tool is appropriate as it offers metrics for measuring the relevance and accuracy of retrieved and generated content.

g. User Interface – The user interface is yet to be determined, as it depends on the feasibility of displaying images within the chat interface. One potential suggestion is using a simple web-based UI built with Streamlit for rapid prototyping and easy integration with the backend, providing a clean and responsive interface for querying and displaying results.

## Task 3:
You are an AI Systems Engineer.  The AI Solutions Engineer has handed off the plan to you.  Now you must identify some source data that you can use for your application.  
Deliverables

1. `Data Sources and External APIs:`

    - Tavily Search API with Google Integration: This will be used to perform top-k searches from Google, retrieving the most relevant and up-to-date splinting information, research articles, and best practices from the web.
    - Indian Journal of Occupational Therapy (IJOT): This source will provide peer-reviewed articles and research papers on OT splinting techniques, ensuring high-quality and reliable data for the chatbot.

2. `Default Chunking Strategy:` 
    - RecursiveCharacterTextSplitter with chunk_size = 500 and chunk_overlap = 0: This method is chosen because it is commonly used in class projects for its simplicity and effectiveness in maintaining coherent text segments. This strategy ensures that text chunks are small enough for processing by the LLM while minimizing redundant overlap.  
    
    `However we will most likely be looking into either:`
    - Paragraph-based Chunking: This approach could be more suitable for preserving the logical flow of information, especially in academic articles, ensuring that related concepts are kept together within the same chunk.
    - Semantic Chunking: Using embeddings to split text based on semantic similarity could help improve the accuracy of retrieved information by keeping related topics within the same chunk.


## Task 4: 

`Prototype:` 
https://huggingface.co/spaces/shivXy/otmidterm
the app.py file will have all the code for the project.  Embeddings will be a different file.

## Task 5: 
1. `Assess pipelines using RAGAS`:
- I added a toggle flag for this because I don’t think evaluations should be running constantly in a production pipeline. Logging the questions and corresponding metrics based on the LLM’s responses makes sense, but evaluating against just the vector store isn’t very useful. Since it's a persistent store, changing the embedding type after encoding the data would be a major issue.  For this project I really wanted to learn vector databases and I feel we are glossing over it in lue of fast prototyping so this is the first time I worked really with a vector store.
` PS C:\_temp\otmidterm> chainlit run app.py
2025-02-25 10:27:28 - Loaded .env file
2025-02-25 10:27:32 - HTTP Request: GET https://40c458f2-24a9-4153-b15b-0addf6a6bbcf.us-east-1-0.aws.cloud.qdrant.io:6333 "HTTP/1.1 200 OK"
2025-02-25 10:27:32 - Your app is available at http://localhost:8000
2025-02-25 10:27:34 - Translated markdown file for en-US not found. Defaulting to chainlit.md.
2025-02-25 10:27:42 - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
🔍 Querying Qdrant with vector: [-0.018138321116566658, 0.03289082273840904, 0.03644593060016632, 0.02279382199048996, 0.01515154354274273]...
2025-02-25 10:27:43 - HTTP Request: POST https://40c458f2-24a9-4153-b15b-0addf6a6bbcf.us-east-1-0.aws.cloud.qdrant.io:6333/collections/qt_document_collection/points/search "HTTP/1.1 200 OK"
🛠 Raw Qdrant Response: [ScoredPoint(id='c598df64-e027-198d-6463-5d1b8e576bd6', version=2, score=0.7267973, payload={'document_name': 'Tennis elbow graded exercise.pdf', 'document_id': '85152085-8c16-4fe9-9302-ad94f260364e', 'document_date': '2025-02-24', 'title': 'Chronic lateral elbow tendinopathy with a supervised graded exercise protocol', 'chunk_number': 1, 'description': 'No description Found', 'author': 'Arzu Razak &#x00D6;zdin&#x00E7;ler PT, PhD', 'tags': ['tag1', 'tag2', 'tag3'], 'subject': 'Journal of Hand Therapy, 36 (2023) 913-922. doi:10.1016/j.jht.2022.11.005'}, vector=None, shard_key=None, order_value=None)]
✅ Filtered Hits: [{'score': 0.7267973, 'metadata': {'document_name': 'Tennis elbow graded exercise.pdf', 'document_id': '85152085-8c16-4fe9-9302-ad94f260364e', 'document_date': '2025-02-24', 'title': 'Chronic lateral elbow tendinopathy with a supervised graded exercise protocol', 'chunk_number': 1, 'description': 'No description Found', 'author': 'Arzu Razak &#x00D6;zdin&#x00E7;ler PT, PhD', 'tags': ['tag1', 'tag2', 'tag3'], 'subject': 'Journal of Hand Therapy, 36 (2023) 913-922. doi:10.1016/j.jht.2022.11.005'}}]
2025-02-25 10:27:49 - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
🔍 Debug: RAGAS Docs Format: []
⚠️ No relevant documents to evaluate.
📊 [Evaluation Mode] RAGAS Score: 0
⚠️ No relevant documents to evaluate.
📊 [evaluate_ragas_metrics] RAGAS Scores: {'faithfulness': 0, 'context_precision': 0, 'context_recall': 0}
2025-02-25 10:27:50 - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-02-25 10:27:51 - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-02-25 10:27:51 - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-02-25 10:27:52 - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"

📊 **Evaluation Results**
                                                      Question                                           Expected Answer                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
                                                                                              Model Answer  Similarity Score
                 What are the best exercises for tennis elbow?       Eccentric exercises and stretching reduce symptoms. **Summary for Occupational Therapists on Lateral Elbow Tendinopathy Treatment**\n\nLateral elbow tendinopathy, affecting 1%-3% of the population, is characterized by pain around the lateral epicondyle and is often a degenerative rather than inflammatory condition. The term 'lateral epicondylitis' is being replaced with 'lateral elbow tendinopathy' to reflect this understanding.\n\nTreatment should focus on a structured exercise program incorporating range-of-motion (ROM) exercises, stretching, and both concentric and eccentric strengthening. Current evidence suggests eccentric exercises, particularly within a heavy slow resistance framework, may yield better outcomes for pain management and functional recovery.\n\nA supervised graded exercise model is recommended, which adjusts based on pain tolerance and healing progress. The approach includes:\n1. **Basic Program (4 weeks)**: Warm-up, isometric strengthening, and functional exercises three times weekly, with a focus on avoiding pain-related inactivity.\n2. **Advanced Program (4 weeks)**: Following an initial assessment of pain levels, this includes stretching and isotonic strengthening exercises.\n\nOutcome measurements should evaluate pain (Visual Analog Scale), functional status (Patient Rated Tennis Elbow Evaluation), and grip strength, assessing changes from baseline after both the basic and advanced exercise periods.\n\nThis structured and tailored approach helps manage lateral elbow tendinopathy effectively, promoting recovery while minimizing pain and functional decline.              0.45
What is the role of manual therapy in treating elbow injuries? Manual therapy improves range of motion and reduces pain. **Summary for Occupational Therapists on Lateral Elbow Tendinopathy Treatment**\n\nLateral elbow tendinopathy, affecting 1%-3% of the population, is characterized by pain around the lateral epicondyle and is often a degenerative rather than inflammatory condition. The term 'lateral epicondylitis' is being replaced with 'lateral elbow tendinopathy' to reflect this understanding.\n\nTreatment should focus on a structured exercise program incorporating range-of-motion (ROM) exercises, stretching, and both concentric and eccentric strengthening. Current evidence suggests eccentric exercises, particularly within a heavy slow resistance framework, may yield better outcomes for pain management and functional recovery.\n\nA supervised graded exercise model is recommended, which adjusts based on pain tolerance and healing progress. The approach includes:\n1. **Basic Program (4 weeks)**: Warm-up, isometric strengthening, and functional exercises three times weekly, with a focus on avoiding pain-related inactivity.\n2. **Advanced Program (4 weeks)**: Following an initial assessment of pain levels, this includes stretching and isotonic strengthening exercises.\n\nOutcome measurements should evaluate pain (Visual Analog Scale), functional status (Patient Rated Tennis Elbow Evaluation), and grip strength, assessing changes from baseline after both the basic and advanced exercise periods.\n\nThis structured and tailored approach helps manage lateral elbow tendinopathy effectively, promoting recovery while minimizing pain and functional decline.              0.47
2025-02-25 10:27:55 - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1740504494.158842    2656 init.cc:232] grpc_wait_for_shutdown_with_timeout() timed out.


2. `Conclusions`: 

- RAGAS seems great for evaluating different prompts, especially during the initial build of a project. However, once you move beyond an in-memory database, modifying embeddings becomes a challenge. Since I'm using Qdrant Cloud, I would need to continuously reload my data whenever embeddings change, which isn't practical at scale. If you have a large dataset, managing this process efficiently would be difficult. While I understand the purpose of in-memory databases, I don't think most organizations would run production systems that way.

## Task 6:  
- here is the link for my embedding model on HF: https://huggingface.co/spaces/shivXy/otmidterm  The code to create the embedding model is located under the fine tuning project.  It has two files one that creates the datasets from the data and the other that creates the model using the data from the jsonl file that were created from the first step.
val_cosine_mrr@10': 0.9772727272727273, 'eval_cosine_map@100': np.float64(0.9772727272727273), 'eval_runtime': 2.4199, 'eval_samples_per_second': 0.0, 'eval_steps_per_second': 0.0, 'epoch': 9.76}
{'train_runtime': 7547.7103, 'train_samples_per_second': 1.076, 'train_steps_per_second': 0.109, 'train_loss': 0.4309439752160049, 'epoch': 10.0}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 820/820 [2:05:47<00:00,  9.20s/it]
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.34G/1.34G [00:53<00:00, 25.1MB/s]

- results from adding the is a bust after waiting 2 hours for the model to process.  when I add it to the embeddings the Qdrant model was trained on a different embedding model.
{"status":{"error":"Wrong input: Vector dimension error: expected dim: 1536, got 1024"},"time":0.000353145}'
code changes to add the HF model to the app:
access_token = os.getenv("HUGGING_FACE_TOKEN")

# Load OpenAI Model
llm = ChatOpenAI(model="gpt-4o-mini")
model = SentenceTransformer("shivXy/ot-midterm-v0")


EVALUATION_MODE = os.getenv("EVALUATION_MODE", "false").lower() == "true"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


## Task 7: 

I'm going to combine both answers in this.  My real tuning came from when I started embedding and sending the data to the qdrant store.  My first application I would output the scores and figure out the best score result from the human message and the result from the cosine search.  So this wasn’t my first attempt at building an application for this task. I wanted to create something that could be useful in a production system, making it as solid as possible, and use that as a discussion point with the group.
I was able to get the application running on my machine and inside a Docker container, but I couldn't get it to work on Hugging Face for a variety of reasons. While I found this assignment valuable, I feel that one week was far too little time—especially since I spent over 40 hours on this and still felt unprepared for the real-world challenges.
I do appreciate having example code to reference, but I feel that most people are just modifying those examples rather than building something new. At the last minute, I had to do the same, which was frustrating. The course might benefit from less slow-paced material early on and more hands-on, structured project-building where everyone develops something from scratch.
This was the best I could do given the time constraints. I had everything working in Docker by yesterday, but by 8 PM, I realized Hugging Face wouldn’t support running my project. I ended up pulling an all-nighter trying to redo it. I learned a lot of valuable concepts through this process—things I wish I had known before starting the project.  With that said my eval results are listed above for this project.  This application is only the app.py file (another issue with hugging face multiple py files).  But in this repo i will include all the work i did for the the folder will be called: Real Project I was building it for my wife and she was the one doing UAT on the results and helped me fine tune the prompts.