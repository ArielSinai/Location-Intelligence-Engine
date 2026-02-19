# Location Intelligence Engine

An intelligent system that helps users decide where to open a new business inside a specific geographic area. The system analyses how crowded each area is with similar services and identifies zones with low competition and high opportunity.

Built as a final project for the Advanced Software Development course, with a focus on two advanced technologies: **Graph Databases** and **Embeddings**.

---

## How It Works

1. The user provides a geographic bounding box and a business type (e.g. "pilates studio")
2. The system collects real POI data from OpenStreetMap for that area
3. Each business is embedded using a sentence-transformer model
4. A Neo4j graph stores businesses, categories, and area relationships
5. Each H3 hexagonal cell is scored on four factors:
   - **Competition density** — fewer direct competitors → better
   - **Demand proxy** — more overall activity → better
   - **Service diversity** — more varied services nearby → better
   - **Semantic saturation** — fewer conceptually similar businesses → better
6. The API returns the top recommended zones with explanation text

---

## Tech Stack

| Layer | Technology |
|---|---|
| POI collection | OpenStreetMap via Overpass API |
| Spatial indexing | H3 hexagonal grid (Uber) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Graph database | Neo4j 5 |
| API | FastAPI + Uvicorn |
| Deployment | Docker / Azure Container |

---

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for Neo4j + API)
- 2 GB free disk space (embedding model download)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Neo4j password:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

### 3. Start Neo4j (and the API) with Docker

```bash
docker-compose up
```

This starts:
- Neo4j at `http://localhost:7474` (browser UI) and `bolt://localhost:7687`
- The API at `http://localhost:8000`

---

## Data Pipeline

Run the three pipeline steps once before using the API. Each step's output feeds the next.

### Step 1 — Collect POIs

Fetches businesses from OpenStreetMap for the chosen area.

```bash
python -m scripts.run_collector \
  --area tel_aviv \
  --south 32.05 --west 34.76 --north 32.10 --east 34.82
```

Output: `data/raw/tel_aviv.json`, `data/raw/tel_aviv_normalized.json`

### Step 2 — Generate embeddings

Runs locally (CPU-intensive — do not run on the free Azure tier).

```bash
python -m scripts.run_embeddings \
  --input data/raw/tel_aviv_normalized.json
```

Output: `data/embeddings/embeddings.npz`

### Step 3 — Build the graph

Populates Neo4j with Business, Category, and Area nodes plus all relationships.

```bash
python -m scripts.run_graph_builder \
  --input data/raw/tel_aviv_normalized.json
```

Add `--clear` to wipe the database before rebuilding.

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

### Get recommendations

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {
      "south": 32.05,
      "west":  34.76,
      "north": 32.10,
      "east":  34.82
    },
    "query": "pilates studio",
    "top_k": 5
  }'
```

**Example response:**

```json
{
  "query": "pilates studio",
  "matched_category": "fitness center",
  "recommendations": [
    {
      "h3_cell": "882db0cc39fffff",
      "rank": 1,
      "opportunity_score": 0.82,
      "competition_count": 1,
      "competition_density": 1.35,
      "demand_proxy": 0.6,
      "diversity_score": 0.7,
      "similarity_saturation": 0.1,
      "support_poi_summary": { "university": 1, "office": 2 },
      "explanation": "Found 1 similar business(es) (1.4 per km²). High activity area — strong foot traffic and demand potential. The area has a diverse mix of complementary services nearby. Low semantic overlap — few conceptually similar businesses present. Located near 2 offices, 1 university, which increases potential customer traffic. Overall opportunity score: 82%."
    }
  ]
}
```

### Request parameters

| Field | Type | Description |
|---|---|---|
| `bbox` | object | `south`, `west`, `north`, `east` in decimal degrees |
| `query` | string | Business type you want to open |
| `top_k` | integer | Number of recommendations to return (default: 5) |

---

## Graph Schema

```
Nodes:
  Business  {osm_id, name, lat, lon, category, h3_cell}
  Category  {name, group}
  Area      {h3_index}

Relationships:
  (Business)-[:BELONGS_TO]->(Category)
  (Business)-[:LOCATED_IN]->(Area)
  (Area)-[:NEAR]->(Area)              — 6 H3 neighbours
  (Area)-[:CLOSE_TO]->(Business)      — demand anchors (universities, offices, transport)
```

---

## Project Structure

```
├── data_collector/      # OSM POI fetching and normalisation
├── embeddings/          # Sentence-transformer wrapper and similarity search
├── graph/               # Neo4j builder and read queries
├── analyzer/            # Opportunity scoring algorithm
├── api/                 # FastAPI app and recommendation endpoint
├── scripts/             # Pipeline runner scripts (steps 1-3)
├── config/              # Pydantic settings
├── tests/               # Full test suite (222 tests)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use mocked external dependencies — no Neo4j connection or internet access required.

---

## Azure Deployment

The API (`api/`) is containerised and intended for Azure Container Instances or Azure App Service.
The three pipeline steps (collect, embed, graph build) are CPU-heavy and should be run locally before deployment.

Build and push the image:

```bash
docker build -t location-intelligence-engine .
docker tag location-intelligence-engine <your-registry>.azurecr.io/location-intelligence-engine
docker push <your-registry>.azurecr.io/location-intelligence-engine
```
