# Multi-Agent Event Prediction System

A comprehensive event prediction system using a 6-agent architecture with multi-policy predictions, consensus ranking, and historical accuracy tracking.

## Architecture

The system consists of six specialized agents:

1. **Query Parser Agent** - Converts natural language to structured queries
2. **Context Retrieval Agent** - Retrieves relevant historical events from TKG
3. **Prediction Agent** - Generates predictions using three independent policies
4. **Ranking Agent** - Ranks and combines predictions with consensus detection
5. **Output Formatter** - Formats results for user consumption
6. **Orchestrator Agent** - Coordinates the entire workflow

## Three Prediction Policies

### 1. Baseline Policy
- **Strategy**: Recency-based predictions
- **Focus**: Recent trends and patterns (2x prediction horizon lookback)
- **Best for**: Short-term predictions, trending events

### 2. Intensity-Aware Policy
- **Strategy**: High-intensity event cascade detection
- **Focus**: Events with high absolute intensity values
- **Best for**: Conflict escalation, violent events, crisis prediction

### 3. Actor History Policy
- **Strategy**: Long-term behavioral pattern analysis
- **Focus**: Complete actor history from 2022 to present
- **Best for**: Predicting consistent actor behaviors, long-term trends

## Quick Start

### Installation

```bash
pip install networkx pandas python-dotenv openai pydantic
```

### Basic Usage

```python
from multi_agent_system import load_data, create_orchestrator

# 1. Load data once (slow operation)
TKG, event_list = load_data(
    csv_path="data/IND_train.csv",
    tkg_path="tkg_memory_loc2.gml"
)

# 2. Create orchestrator once
orchestrator = create_orchestrator(tkg=TKG, event_list=event_list)

# 3. Run multiple queries (fast)
result = orchestrator.orchestrate(
    user_query="What will Narendra Modi do in the next month in UP?",
    prediction_horizon_days=30,
    output_format="text",
    verbose=True
)

print(result)
```

## API Reference

### `load_data(csv_path, tkg_path)`

Load TKG and event list from files.

**Parameters:**
- `csv_path` (str): Path to CSV file with event data
- `tkg_path` (str): Path to GML file with TKG

**Returns:**
- Tuple of (TKG graph, event list)

### `create_orchestrator(tkg, event_list)`

Create an orchestrator agent with pre-loaded data.

**Parameters:**
- `tkg` (nx.DiGraph): Pre-loaded Temporal Knowledge Graph
- `event_list` (List[CausalMemoryEvent]): Pre-loaded event list

**Returns:**
- Initialized OrchestratorAgent

### `orchestrator.orchestrate(user_query, prediction_horizon_days, output_format, verbose)`

Execute the complete prediction workflow.

**Parameters:**
- `user_query` (str): Natural language query (e.g., "Will there be protests in Delhi?")
- `prediction_horizon_days` (int, default=30): Days ahead to predict
- `output_format` (str, default="text"): Output format ("text" or "json")
- `verbose` (bool, default=False): If True, show all 3 policy predictions

**Returns:**
- Formatted prediction results (string)

## Output Modes

### Verbose Mode (`verbose=True`)

Shows detailed breakdown:

```
INDIVIDUAL POLICY PREDICTIONS
================================================================================

POLICY #1: BASELINE
--------------------------------------------------------------------------------
Event Type: Accuse
Raw Confidence: 85.0%
Reasoning: Recent events show a pattern of accusations...
Supporting Events: [event IDs]

POLICY #2: INTENSITY_AWARE
--------------------------------------------------------------------------------
...

POLICY #3: ACTOR_HISTORY
--------------------------------------------------------------------------------
...

FINAL RANKED PREDICTIONS
================================================================================
PREDICTION #1 (Confidence: 79.2%)
...
```

### Concise Mode (`verbose=False`)

Shows only final ranked predictions:

```
FINAL RANKED PREDICTIONS
================================================================================
PREDICTION #1 (Confidence: 79.2%)
--------------------------------------------------------------------------------
Event Type: Accuse
Contributing Policies: baseline, intensity_aware, actor_history
Consensus Bonus: +20.0%
```

## Confidence Scoring

Confidence scores are calculated based on:

1. **Historical Accuracy**: Each policy's past performance
   - Overall accuracy (40% weight)
   - Event-type-specific accuracy (30% weight)
   - Horizon-specific accuracy (30% weight)

2. **Policy Raw Confidence**: LLM's confidence (0-1) applied as multiplier

3. **Consensus Bonus**: +10% per additional agreeing policy
   - 1 policy: base confidence
   - 2 policies: +10% bonus
   - 3 policies: +20% bonus

## Historical Accuracy Database

Default accuracy rates:

| Policy | Overall | Protest | Conflict | Negotiation |
|--------|---------|---------|----------|-------------|
| Baseline | 65% | 72% | 58% | 70% |
| Intensity-Aware | 58% | 55% | 68% | 45% |
| Actor History | 70% | 68% | 71% | 75% |

## Example Queries

```python
# Short-term prediction
orchestrator.orchestrate(
    "What will happen in Kashmir next week?",
    prediction_horizon_days=7
)

# Actor-specific query
orchestrator.orchestrate(
    "What will Narendra Modi do in the next month?",
    prediction_horizon_days=30
)

# Location-specific query
orchestrator.orchestrate(
    "Will there be protests in Delhi?",
    prediction_horizon_days=14
)

# Event-type query
orchestrator.orchestrate(
    "Will India and Pakistan negotiate soon?",
    prediction_horizon_days=60
)
```

## JSON Output Format

For API integration, use `output_format="json"`:

```json
[
  {
    "prediction_id": "uuid",
    "combined_prediction": {
      "event_type": "Accuse",
      "predicted_timestamp": "2025-12-09T00:00:00",
      "actors": ["Narendra Modi", "Opposition"],
      "location": "UP",
      "intensity": -3.0,
      "description": "..."
    },
    "confidence_score": 0.792,
    "contributing_policies": ["baseline", "intensity_aware", "actor_history"],
    "policy_confidences": {
      "baseline": 0.5346,
      "intensity_aware": 0.4981,
      "actor_history": 0.5924
    },
    "consensus_bonus": 0.2,
    "explanation": "..."
  }
]
```

## Performance Considerations

- **Initial Load**: Loading TKG and events takes ~5-10 seconds
- **Per Query**: Each prediction takes ~10-30 seconds (3 LLM calls)
- **Optimization**: Load data once, run multiple queries on same orchestrator

## Requirements

- Python 3.8+
- OpenAI API key (set in `.env` file as `OPENAI_API_KEY`)
- Pre-built TKG (GML format)
- Event data (CSV format)

## File Structure

```
TMS/
├── multi_agent_system.py       # Main system implementation
├── example_usage.py            # Usage examples
├── tkg_builder.py              # TKG creation utilities
├── memory_module.py            # Context retrieval functions
├── data/
│   └── IND_train.csv          # Event data
└── tkg_memory_loc2.gml        # Pre-built TKG
```

## Troubleshooting

### "No events found matching criteria"
- Check that location codes match (use normalized codes without backslashes)
- Verify actor names match those in the dataset
- Try broader queries with fewer constraints

### "Prediction failed"
- Check OpenAI API key is set in `.env`
- Verify API quota/rate limits
- Check network connectivity

### "TypeError: Cannot compare tz-naive and tz-aware timestamps"
- This should be handled automatically in the code
- If it persists, check pandas version compatibility

## License

MIT License

## Contact

For questions or issues, please open an issue on the GitHub repository.
