# Performance Metrics Documentation

## Overview

The Multi-Agent Event Prediction System now includes comprehensive performance metrics to evaluate prediction quality against actual historical events. This allows you to quantitatively assess how well the system predicts future events.

## Available Metrics

### 1. Hit@k (Hit Rate at k)

**Definition**: Measures whether any actual event type appears in the top-k predictions.

**Formula**: Hit@k = 1 if actual event type ∈ top-k predictions, else 0

**Interpretation**:
- `Hit@1 = 1.0`: The top prediction matches an actual event type (perfect!)
- `Hit@3 = 1.0`: At least one of the top 3 predictions matches reality
- `Hit@5 = 1.0`: At least one of the top 5 predictions matches reality

**Example**:
```
Actual events: Accuse, Protest
Predicted (ranked): Accuse, Response, Sanction

Hit@1 = 1.0  (Accuse matches)
Hit@3 = 1.0  (Accuse matches)
```

### 2. MRR (Mean Reciprocal Rank)

**Definition**: Measures the ranking quality by looking at the position of the first correct prediction.

**Formula**: MRR = 1 / rank of first correct prediction

**Interpretation**:
- `MRR = 1.0`: First prediction is correct (rank 1)
- `MRR = 0.5`: First correct prediction is at rank 2
- `MRR = 0.333`: First correct prediction is at rank 3
- `MRR = 0.0`: No correct predictions

**Example**:
```
Actual events: Protest
Predicted (ranked): Accuse, Response, Protest

MRR = 1/3 = 0.333 (first match at rank 3)
```

### 3. Type Accuracy

**Definition**: Exact match of the top prediction with actual event types.

**Formula**: Type Accuracy = 1 if top prediction matches any actual event type, else 0

**Interpretation**:
- `1.0`: Top prediction is correct
- `0.0`: Top prediction doesn't match reality

## Usage

### Basic Usage

```python
from multi_agent_system import load_data, create_orchestrator
from performance_metrics import PredictionMetrics

# Load data
TKG, event_list = load_data(csv_path="...", tkg_path="...")
orchestrator = create_orchestrator(tkg=TKG, event_list=event_list)
metrics_evaluator = PredictionMetrics(event_list)

# Run prediction with metrics
output, predictions, parsed_query = orchestrator.orchestrate_with_metrics(
    user_query="Rahul Gandhi accused Narendra Modi on June 26, 2022. What followed in 2 months?",
    prediction_horizon_days=60,
    verbose=True
)

# Evaluate
metrics = metrics_evaluator.evaluate_prediction(
    predictions=predictions,
    reference_date="2022-06-26",
    horizon_days=60,
    actor="Rahul Gandhi",  # Optional filter
    location="DL",         # Optional filter
    k_values=[1, 3, 5]     # Evaluate Hit@1, Hit@3, Hit@5
)

# Display report
print(metrics_evaluator.format_metrics_report(metrics))
```

### Output Example

```
================================================================================
PERFORMANCE METRICS
================================================================================

Actual Events in Horizon: 4
Number of Predictions: 1

Actual Event Types: Accuse, Protest, Response
Predicted Event Types (top 5): Accuse

--------------------------------------------------------------------------------
HIT@K METRICS (Does any actual event appear in top-k predictions?)
--------------------------------------------------------------------------------
Hit@1: 1.00 [HIT]
  -> Matched types: accuse
Hit@3: 1.00 [HIT]
  -> Matched types: accuse
Hit@5: 1.00 [HIT]
  -> Matched types: accuse

--------------------------------------------------------------------------------
MEAN RECIPROCAL RANK (MRR)
--------------------------------------------------------------------------------
MRR: 1.0000 (first match at rank 1)

--------------------------------------------------------------------------------
TYPE ACCURACY (Top prediction exact match)
--------------------------------------------------------------------------------
Accuracy: 1.00 [CORRECT]
  -> Matched: accuse
================================================================================
```

## Metrics Dictionary Structure

The `evaluate_prediction()` method returns a dictionary with the following structure:

```python
{
    'num_actual_events': 4,           # Number of actual events in horizon
    'num_predictions': 3,             # Number of predictions made
    'actual_event_types': ['Accuse', 'Protest'],  # Actual event types
    'predicted_event_types': ['Accuse', 'Response', 'Sanction'],  # Top 5 predictions

    # Hit@k metrics
    'hit@1': 1.0,                     # Hit rate at k=1
    'hit@1_matches': ['accuse'],      # Matched types for k=1
    'hit@3': 1.0,                     # Hit rate at k=3
    'hit@3_matches': ['accuse'],      # Matched types for k=3
    'hit@5': 1.0,                     # Hit rate at k=5
    'hit@5_matches': ['accuse'],      # Matched types for k=5

    # MRR metrics
    'mrr': 1.0,                       # Mean Reciprocal Rank
    'first_match_rank': 1,            # Rank of first correct prediction

    # Type accuracy
    'type_accuracy': 1.0,             # Top prediction exact match
    'exact_match': ['accuse']         # Matched type(s)
}
```

## Aggregate Metrics

To evaluate performance across multiple queries:

```python
all_metrics = [metrics1, metrics2, metrics3]

# Calculate averages
avg_hit1 = sum(m['hit@1'] for m in all_metrics) / len(all_metrics)
avg_hit3 = sum(m['hit@3'] for m in all_metrics) / len(all_metrics)
avg_mrr = sum(m['mrr'] for m in all_metrics) / len(all_metrics)

print(f"Average Hit@1: {avg_hit1:.2%}")
print(f"Average Hit@3: {avg_hit3:.2%}")
print(f"Average MRR: {avg_mrr:.4f}")
```

## Filtering Actual Events

You can filter actual events by actor and location to focus metrics on specific scenarios:

```python
# Only evaluate events involving "Supreme Court" in "UP"
metrics = metrics_evaluator.evaluate_prediction(
    predictions=predictions,
    reference_date="2022-05-06",
    horizon_days=60,
    actor="Supreme Court",   # Filter by actor
    location="UP"            # Filter by location
)
```

## Interpreting Results

### Query 1 Example (from output)
```
Query 1: Hit@3=0 [N], MRR=0.0000, Actual events=0
```
- **No actual events found** in the prediction horizon (possibly actor/location filters too restrictive)
- Cannot evaluate metrics without ground truth
- Consider broadening filters or checking if events exist

### Query 2 Example
```
Query 2: Hit@3=1 [Y], MRR=1.0000, Actual events=4
```
- **Perfect prediction**: Top prediction matches actual events
- 4 actual events occurred in the horizon
- MRR = 1.0 means first prediction was correct

### Query 3 Example
```
Query 3: Hit@3=1 [Y], MRR=1.0000, Actual events=2
```
- **Perfect prediction**: Top prediction matches actual events
- 2 actual events occurred in the horizon
- MRR = 1.0 means first prediction was correct

### Overall Performance
```
Average Performance Across 3 Queries:
  Hit@1:         66.67%   (2 out of 3 queries had correct top prediction)
  Hit@3:         66.67%   (2 out of 3 queries had match in top 3)
  MRR:           0.6667   (Average ranking quality)
  Type Accuracy: 66.67%   (2 out of 3 top predictions correct)
```

## Best Practices

1. **Use Historical Queries**: For evaluation, use historical event-based queries where you know what actually happened

2. **Set Appropriate Filters**: Use actor/location filters to focus on relevant events:
   ```python
   # Too broad - may include unrelated events
   metrics = evaluate_prediction(..., actor=None, location=None)

   # Better - focused on specific actor/location
   metrics = evaluate_prediction(..., actor="Rahul Gandhi", location="KA")
   ```

3. **Multiple k Values**: Evaluate Hit@1, Hit@3, and Hit@5 to understand:
   - Hit@1: How often is the top prediction correct?
   - Hit@3: How often does the top 3 include the correct answer?
   - Hit@5: Broader coverage

4. **Aggregate Across Queries**: Don't rely on single query metrics - average across multiple queries for robust evaluation

5. **Check Actual Event Count**: If `num_actual_events = 0`, the query might be:
   - Too early (no events happened yet in horizon)
   - Filtered too strictly (actor/location mismatch)
   - Reference event date incorrect

## Files

- **performance_metrics.py**: Core metrics implementation
- **example_usage.py**: Example usage with 3 historical queries
- **example_output_with_metrics.txt**: Sample output showing all metrics

## Future Enhancements

Potential additional metrics:
- **NDCG (Normalized Discounted Cumulative Gain)**: Weighted ranking metric
- **Precision@k**: Proportion of top-k predictions that are correct
- **Recall@k**: Proportion of actual events covered in top-k
- **F1@k**: Harmonic mean of Precision@k and Recall@k
- **Actor/Location Accuracy**: Evaluate predicted actors and locations, not just event types
- **Temporal Accuracy**: How close is the predicted date to actual event dates?
