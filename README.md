# Vector_AI — Vector Database + RAG Engine from Scratch in Java

Zero-dependency **vector database** in pure Java. Three search engines (**HNSW**, **KD-Tree**, **Brute Force**) side by side, a live web UI, and a **RAG pipeline** powered by local LLMs through Ollama.

> Educational build showing how Pinecone, Weaviate, Chroma and Milvus work under the hood.

![Java](https://img.shields.io/badge/Java-17%2B-orange)
![Dependencies](https://img.shields.io/badge/Dependencies-0-brightgreen)
![Ollama](https://img.shields.io/badge/LLM-Ollama-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Request Flow](#request-flow)
4. [Document Ingestion Flow](#document-ingestion-flow)
5. [RAG Pipeline Flow](#rag-pipeline-flow)
6. [Algorithm Flowcharts](#algorithm-flowcharts)
7. [Class Structure](#class-structure)
8. [Tech Stack](#tech-stack)
9. [Setup](#setup)
10. [Usage](#usage)
11. [REST API](#rest-api)
12. [Project Structure](#project-structure)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)

---

## Features

| Feature | Description |
|---|---|
| 3 search algorithms | HNSW, KD-Tree, Brute Force — run all three, compare speed |
| 3 distance metrics | Cosine, Euclidean, Manhattan |
| 16D demo vectors | 20 preloaded vectors across CS, Math, Food, Sports |
| PCA scatter plot | Live 2D projection of semantic space |
| Real embeddings | Any text → `nomic-embed-text` → 768D vector |
| RAG pipeline | Question → HNSW retrieval → `llama3.2` answer |
| REST API | Insert, delete, search, benchmark, hnsw-info, doc ingest, ask |

---

## System Architecture

```mermaid
flowchart TB
    subgraph Client["Browser — index.html"]
        T1["Tab 1: Search"]
        T2["Tab 2: Documents"]
        T3["Tab 3: Ask AI"]
        VIZ["PCA Scatter Plot"]
    end

    subgraph Server["Main.java — HTTP Server :8080"]
        ROUTER["REST Router"]
        subgraph Demo["VectorDB — 16D"]
            BF["BruteForce"]
            KD["KDTree"]
            HN["HNSW"]
        end
        subgraph Docs["DocumentDB — 768D"]
            DH["HNSW Index"]
            CH["Chunker"]
        end
        OC["OllamaClient"]
    end

    subgraph Ollama["Ollama — Local AI"]
        EMB["nomic-embed-text"]
        LLM["llama3.2"]
    end

    T1 --> ROUTER
    T2 --> ROUTER
    T3 --> ROUTER
    ROUTER --> Demo
    ROUTER --> Docs
    Docs --> OC
    OC --> EMB
    OC --> LLM
    ROUTER --> VIZ
```

---

## Request Flow

```mermaid
flowchart TD
    A["HTTP Request"] --> B{"Route?"}
    B -->|"GET /search"| C["Parse v, k, metric, algo"]
    B -->|"GET /benchmark"| D["Run all 3 algorithms"]
    B -->|"POST /insert"| E["Insert into BF + KD + HNSW"]
    B -->|"DELETE /delete/:id"| F["Remove from all indexes"]
    B -->|"POST /doc/insert"| G["Document Ingestion Flow"]
    B -->|"POST /doc/ask"| H["RAG Pipeline Flow"]
    B -->|"GET /status"| I["Ping Ollama"]

    C --> J{"algo?"}
    J -->|"hnsw"| K["HNSW search"]
    J -->|"kdtree"| L["KD-Tree search"]
    J -->|"brute"| M["Brute Force scan"]

    K --> N["Top-K results + latency"]
    L --> N
    M --> N
    D --> O["Per-algorithm time + results"]

    N --> P["JSON Response"]
    O --> P
    E --> P
    F --> P
    I --> P
```

---

## Document Ingestion Flow

```mermaid
flowchart TD
    A["Title + Text"] --> B["POST /doc/insert"]
    B --> C["Split into overlapping 250-word chunks"]
    C --> D["For each chunk"]
    D --> E["OllamaClient.embed"]
    E --> F["nomic-embed-text"]
    F --> G["768D vector"]
    G --> H["DocumentDB.insert"]
    H --> I["HNSW Index"]
    I --> J{"More chunks?"}
    J -->|"Yes"| D
    J -->|"No"| K["Return chunk count + IDs"]
```

---

## RAG Pipeline Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Server
    participant E as nomic-embed-text
    participant H as HNSW Index
    participant L as llama3.2

    U->>S: POST /doc/ask question, k=3
    S->>E: embed question
    E-->>S: 768D query vector
    S->>H: search top-k
    H-->>S: 3 nearest chunks
    S->>S: build prompt = context + question
    S->>L: generate answer
    L-->>S: answer text
    S-->>U: answer + context chips
```

```mermaid
flowchart LR
    Q["Question"] --> EQ["Embed"]
    EQ --> RS["HNSW Search k=3"]
    RS --> CT["Context Chunks"]
    CT --> PR["Prompt Builder"]
    Q --> PR
    PR --> GN["llama3.2"]
    GN --> AN["Answer"]
    CT --> CC["Context Chips in UI"]
```

---

## Algorithm Flowcharts

### HNSW Insert

```mermaid
flowchart TD
    A["New vector q"] --> B["Assign random max layer L"]
    B --> C["Start at entry point, top layer"]
    C --> D{"layer greater than L?"}
    D -->|"Yes"| E["Greedy descent to nearest node"]
    E --> F["Move one layer down"]
    F --> D
    D -->|"No"| G["Beam search with ef_construction = 200"]
    G --> H["Select M nearest neighbors"]
    H --> I["Connect bidirectionally"]
    I --> J["Prune neighbors exceeding M"]
    J --> K{"layer greater than 0?"}
    K -->|"Yes"| L["Move one layer down"]
    L --> G
    K -->|"No"| M{"L above current top?"}
    M -->|"Yes"| N["Update entry point"]
    M -->|"No"| O["Done"]
    N --> O
```

### HNSW Search

```mermaid
flowchart TD
    A["Query vector"] --> B["Entry point at top layer"]
    B --> C{"layer greater than 0?"}
    C -->|"Yes"| D["Greedy move to closer neighbor"]
    D --> E{"Closer node found?"}
    E -->|"Yes"| D
    E -->|"No"| F["Drop one layer"]
    F --> C
    C -->|"No"| G["Layer 0: expand ef candidates with priority queue"]
    G --> H["Keep best ef results"]
    H --> I["Return top-K"]
```

### KD-Tree Search

```mermaid
flowchart TD
    A["Query + current node"] --> B{"Node null?"}
    B -->|"Yes"| Z["Return"]
    B -->|"No"| C["Compute distance to node point"]
    C --> D{"Better than current worst in top-K?"}
    D -->|"Yes"| E["Update best list"]
    D -->|"No"| F["Skip"]
    E --> G["Pick near side by split dimension"]
    F --> G
    G --> H["Recurse near subtree"]
    H --> I{"Hyperplane distance below worst best?"}
    I -->|"Yes"| J["Recurse far subtree"]
    I -->|"No"| K["Prune far subtree"]
    J --> Z
    K --> Z
```

### Brute Force

```mermaid
flowchart LR
    A["Query"] --> B["Compute distance to every vector"]
    B --> C["Sort ascending"]
    C --> D["Return top-K"]
```

### Complexity

| Algorithm | Search | Exact | Best For |
|---|---|---|---|
| Brute Force | O(N·d) | Yes | Baseline, small data |
| KD-Tree | O(log N), degrades in high-D | Yes | Low dimensions (≤20D) |
| HNSW | O(log N) | Approximate | High dimensions, production |

```mermaid
flowchart LR
    A["Dimension"] --> B{"d at most 20?"}
    B -->|"Yes"| C["KD-Tree prunes well"]
    B -->|"No"| D["KD-Tree approaches brute force"]
    D --> E["HNSW graph navigation unaffected"]
```

---

## Class Structure

```mermaid
classDiagram
    class Main {
        +main()
        +startServer()
    }
    class VectorDB {
        +insert()
        +delete()
        +search()
        +benchmark()
    }
    class DocumentDB {
        +insertDocument()
        +askQuestion()
        +list()
        +delete()
    }
    class BruteForce {
        +search()
    }
    class KDTree {
        +insert()
        +search()
    }
    class HNSW {
        +insert()
        +search()
        +info()
    }
    class OllamaClient {
        +embed()
        +generate()
        +status()
        +genModel
    }

    Main --> VectorDB
    Main --> DocumentDB
    VectorDB --> BruteForce
    VectorDB --> KDTree
    VectorDB --> HNSW
    DocumentDB --> HNSW
    DocumentDB --> OllamaClient
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Java 17+, built-in HTTP server, no external libraries |
| Indexes | HNSW, KD-Tree, Brute Force |
| Frontend | HTML, CSS, JavaScript, Canvas (PCA scatter plot) |
| Embeddings | Ollama `nomic-embed-text` (768D) |
| LLM | Ollama `llama3.2` |
| Visualization | PCA 16D → 2D |

---

## Setup

### Prerequisites

| Tool | Version |
|---|---|
| Java JDK | 17+ |
| Git | Any |
| Ollama | Latest, 8 GB RAM recommended |

### 1. Verify Java

```powershell
java -version
javac -version
```

Install from https://adoptium.net/ if missing.

### 2. Install Ollama and Pull Models

Download from https://ollama.com, then:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
ollama list
```

### 3. Clone

```powershell
git clone https://github.com/krishkantrai27/Vector_AI.git
cd Vector_AI
```

### 4. Run

```powershell
javac Main.java
java Main
```

Or directly:

```powershell
java Main.java
```

Expected output:

```
=== VectorDB Engine (Java) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
Server listening on port 8080...
```

Open http://localhost:8080

```mermaid
flowchart LR
    A["Install Java"] --> B["Install Ollama"]
    B --> C["Pull 2 models"]
    C --> D["Clone repo"]
    D --> E["java Main.java"]
    E --> F["Open localhost:8080"]
```

---

## Usage

### Tab 1 — Search

1. Enter a concept: `binary tree`, `sushi`, `basketball`, `calculus`
2. Pick algorithm: HNSW / KD-Tree / Brute Force
3. Pick metric: Cosine / Euclidean / Manhattan
4. Click **SEARCH** — results with distances, match glows on scatter plot
5. Click **COMPARE ALL ALGOS** — speed comparison of all three

### Tab 2 — Documents

1. Enter a title
2. Paste text
3. Click **EMBED & INSERT**
4. Text is split into overlapping 250-word chunks, each embedded and indexed in HNSW

### Tab 3 — Ask AI

1. Insert documents first
2. Type a question
3. Click **ASK AI**
4. Answer streams in; click context chips to inspect retrieved chunks

---

## REST API

Base URL: `http://localhost:8080`

### Demo Vectors

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| POST | `/insert` | Insert demo vector |
| DELETE | `/delete/:id` | Delete by ID |
| GET | `/items` | List all vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Compare 3 algorithms |
| GET | `/hnsw-info` | Graph structure, layer stats |
| GET | `/stats` | Database statistics |

### Documents and RAG

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store |
| GET | `/doc/list` | — | List chunks |
| DELETE | `/doc/delete/:id` | — | Delete chunk |
| POST | `/doc/ask` | `{"question":"...","k":3}` | Retrieve + generate |
| GET | `/status` | — | Ollama status |

### Examples

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
Vector_AI/
├── Main.java       Backend: HNSW, KD-Tree, BruteForce, REST API, RAG
├── index.html      Frontend: PCA plot, chat UI, benchmark
└── README.md
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` |
| First embed very slow | Model loading, wait ~2 min |
| `java: command not found` | Add JDK 17+ to PATH |
| Port 8080 busy | `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| Slow LLM answers | Use `llama3.2:1b` |

Faster model:

```powershell
ollama pull llama3.2:1b
```

In `Main.java`:

```java
public String genModel = "llama3.2:1b";
```

Recompile and restart.

---

## License

MIT
