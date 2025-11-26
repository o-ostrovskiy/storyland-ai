# Why Agents? - Visual Diagrams

## Sequential vs Parallel Execution Comparison

### Option 1: Timeline Comparison (Recommended for Video)

```mermaid
gantt
    title Single LLM (Sequential) - 60 seconds total
    dateFormat X
    axisFormat %L

    section Sequential
    City Research (20s)      :0, 20
    Landmark Search (20s)    :20, 40
    Author Sites (20s)       :40, 60
```

```mermaid
gantt
    title Multi-Agent System (Parallel) - 20 seconds total
    dateFormat X
    axisFormat %L

    section Parallel
    City Agent (20s)         :0, 20
    Landmark Agent (20s)     :0, 20
    Author Agent (20s)       :0, 20
```

### Option 2: Flowchart Comparison

```mermaid
flowchart LR
    subgraph single["❌ Single LLM (Sequential)"]
        direction LR
        A1[City Research<br/>20s] --> A2[Landmark Search<br/>20s] --> A3[Author Sites<br/>20s]
        A3 --> A4[Total: 60s]
    end

    style single fill:#ffe6e6
    style A4 fill:#ff6b6b,color:#fff
```

```mermaid
flowchart TB
    subgraph multi["✅ Multi-Agent System (Parallel)"]
        direction TB
        B0[Book Input] --> B1
        B0 --> B2
        B0 --> B3
        B1[City Agent<br/>20s]
        B2[Landmark Agent<br/>20s]
        B3[Author Agent<br/>20s]
        B1 --> B4[Total: 20s<br/>3x FASTER!]
        B2 --> B4
        B3 --> B4
    end

    style multi fill:#e6ffe6
    style B4 fill:#51cf66,color:#fff
```

### Option 3: Side-by-Side Feature Comparison

```mermaid
flowchart LR
    subgraph comparison[" "]
        direction TB

        subgraph left["Single LLM"]
            L1["❌ Sequential execution"]
            L2["❌ 60 seconds"]
            L3["❌ Limited context"]
            L4["❌ Generic results"]
            L5["❌ Hallucination risk"]
        end

        subgraph right["Multi-Agent System"]
            R1["✅ Parallel execution"]
            R2["✅ 20 seconds (3x faster)"]
            R3["✅ Specialized agents"]
            R4["✅ Coordinated intelligence"]
            R5["✅ Real data sources"]
        end
    end

    style left fill:#ffe6e6
    style right fill:#e6ffe6
    style comparison fill:#f8f9fa
```

### Option 4: Agent Architecture Overview

```mermaid
flowchart TB
    START[Book Input:<br/>The Great Gatsby]

    subgraph parallel["Parallel Discovery (20s)"]
        direction LR
        A1[City Agent<br/>🏙️<br/>New York, Long Island]
        A2[Landmark Agent<br/>🏛️<br/>Oheka Castle, Plaza Hotel]
        A3[Author Agent<br/>✍️<br/>Fitzgerald homes, museums]
    end

    START --> parallel
    parallel --> ANALYZE[Region Analyzer<br/>Group by geography]
    ANALYZE --> RESULT[3x faster than<br/>sequential execution!]

    style parallel fill:#e6f3ff
    style RESULT fill:#51cf66,color:#fff
```

### Option 5: Full 6-Agent System (Extended View)

```mermaid
flowchart TB
    INPUT[📚 Book Input]

    INPUT --> META[1. Metadata Agent<br/>Find exact book<br/>10s]

    META --> CONTEXT[2. Context Agent<br/>Research setting<br/>10s]

    CONTEXT --> PARALLEL

    subgraph PARALLEL["3. Parallel Discovery (20s) ⚡"]
        direction LR
        CITY[City Agent<br/>🏙️]
        LAND[Landmark Agent<br/>🏛️]
        AUTH[Author Agent<br/>✍️]
    end

    PARALLEL --> REGION[4. Region Analyzer<br/>Group locations<br/>10s]

    REGION --> HITL{👤 Human Selects<br/>Region}

    HITL --> COMPOSE[5. Trip Composer<br/>Personalized itinerary<br/>30s]

    COMPOSE --> OUTPUT[📍 Final Itinerary]

    style PARALLEL fill:#e6f3ff
    style HITL fill:#fff3bf
    style OUTPUT fill:#51cf66,color:#fff
```

### Option 6: Simple Speed Comparison Bar Chart

```mermaid
graph LR
    subgraph speed["Execution Time Comparison"]
        direction TB

        A["Single LLM<br/>███████████████████████████████████████ 60s"]
        B["Multi-Agent<br/>█████████████ 20s"]
        C["Speedup: 3x FASTER! ⚡"]
    end

    style A fill:#ff6b6b,color:#fff
    style B fill:#51cf66,color:#fff
    style C fill:#ffd43b
```