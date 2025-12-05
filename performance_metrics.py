"""
Performance Metrics for Event Prediction Evaluation

This module provides metrics to evaluate the quality of event predictions:
- Hit@k: Measures if the actual event appears in top-k predictions
- MRR (Mean Reciprocal Rank): Measures ranking quality
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
from tkg_builder import *
from multi_agent_system import *
import networkx as nx


class PredictionMetrics:
    """Calculate performance metrics for event predictions."""

    def __init__(self, event_list):
        """
        Initialize metrics calculator.

        Args:
            event_list: List of CausalMemoryEvent objects
        """
        self.event_list = event_list
        self.event_dict = {e.event_id: e for e in event_list}

    def get_actual_events(
        self,
        reference_date: str,
        horizon_days: int,
        actor: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get actual events that occurred within the prediction horizon.

        Args:
            reference_date: Starting date (YYYY-MM-DD)
            horizon_days: Number of days to look ahead
            actor: Optional actor filter
            location: Optional location filter

        Returns:
            List of actual events with their details
        """
        start_date = pd.to_datetime(reference_date)
        end_date = start_date + timedelta(days=horizon_days)

        actual_events = []
        for event in self.event_list:
            event_date = pd.to_datetime(event.time_anchor)

            # Check if event is within horizon
            if start_date < event_date <= end_date:
                # Apply filters if specified
                if actor and actor.lower() not in event.actor_subj.lower() and \
                   actor.lower() not in event.recipient_obj.lower():
                    continue

                if location and location not in event.obj_state:
                    continue

                actual_events.append({
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'actor': event.actor_subj,
                    'recipient': event.recipient_obj,
                    'location': event.obj_state,
                    'date': event.time_anchor,
                    'intensity': event.intensity
                })

        return actual_events

    def calculate_hit_at_k(
        self,
        predictions: List[Dict[str, Any]],
        actual_events: List[Dict[str, Any]],
        k: int = 3
    ) -> Tuple[float, List[str]]:
        """
        Calculate Hit@k metric.

        Hit@k = 1 if any actual event type appears in top-k predictions, else 0

        Args:
            predictions: List of predicted events (ranked by confidence)
            actual_events: List of actual events that occurred
            k: Number of top predictions to consider

        Returns:
            (hit_score, matched_event_types)
        """
        if not actual_events:
            return 0.0, []

        # Get top-k predicted event types
        top_k_predictions = predictions[:k]
        predicted_types = set(p['event_type'].lower() for p in top_k_predictions)

        # Get actual event types
        actual_types = set(e['event_type'].lower() for e in actual_events)

        # Check for matches
        matches = predicted_types.intersection(actual_types)

        hit_score = 1.0 if matches else 0.0
        return hit_score, list(matches)

    def calculate_mrr(
        self,
        predictions: List[Dict[str, Any]],
        actual_events: List[Dict[str, Any]]
    ) -> Tuple[float, Optional[int]]:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / rank of first correct prediction
        where rank starts at 1

        Args:
            predictions: List of predicted events (ranked by confidence)
            actual_events: List of actual events that occurred

        Returns:
            (mrr_score, first_match_rank)
        """
        if not actual_events:
            return 0.0, None

        # Get actual event types
        actual_types = set(e['event_type'].lower() for e in actual_events)

        # Find rank of first matching prediction
        for rank, prediction in enumerate(predictions, start=1):
            if prediction['event_type'].lower() in actual_types:
                mrr_score = 1.0 / rank
                return mrr_score, rank

        # No match found
        return 0.0, None

    def calculate_type_accuracy(
        self,
        predictions: List[Dict[str, Any]],
        actual_events: List[Dict[str, Any]]
    ) -> Tuple[float, List[str]]:
        """
        Calculate type accuracy - exact match of top prediction.

        Args:
            predictions: List of predicted events (ranked by confidence)
            actual_events: List of actual events that occurred

        Returns:
            (accuracy, matched_types)
        """
        if not predictions or not actual_events:
            return 0.0, []

        top_prediction_type = predictions[0]['event_type'].lower()
        actual_types = set(e['event_type'].lower() for e in actual_events)

        if top_prediction_type in actual_types:
            return 1.0, [top_prediction_type]

        return 0.0, []

    def evaluate_prediction(
        self,
        predictions: List[Dict[str, Any]],
        reference_date: str,
        horizon_days: int,
        actor: Optional[str] = None,
        location: Optional[str] = None,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of predictions.

        Args:
            predictions: List of predicted events
            reference_date: Starting date for predictions
            horizon_days: Prediction horizon
            actor: Optional actor filter for actual events
            location: Optional location filter
            k_values: List of k values for Hit@k metric

        Returns:
            Dictionary with all metrics
        """
        # Get actual events
        actual_events = self.get_actual_events(
            reference_date, horizon_days, actor, location
        )

        # Calculate metrics
        results = {
            'num_actual_events': len(actual_events),
            'num_predictions': len(predictions),
            'actual_event_types': list(set(e['event_type'] for e in actual_events)),
            'predicted_event_types': [p['event_type'] for p in predictions[:5]],
        }

        # Hit@k for different k values
        for k in k_values:
            hit_score, matches = self.calculate_hit_at_k(predictions, actual_events, k)
            results[f'hit@{k}'] = hit_score
            results[f'hit@{k}_matches'] = matches

        # MRR
        mrr_score, first_rank = self.calculate_mrr(predictions, actual_events)
        results['mrr'] = mrr_score
        results['first_match_rank'] = first_rank

        # Type accuracy
        type_acc, matched = self.calculate_type_accuracy(predictions, actual_events)
        results['type_accuracy'] = type_acc
        results['exact_match'] = matched

        return results

    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics into a readable report.

        Args:
            metrics: Dictionary from evaluate_prediction()

        Returns:
            Formatted string report
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("PERFORMANCE METRICS")
        report.append("=" * 80)

        # Basic info
        report.append(f"\nActual Events in Horizon: {metrics['num_actual_events']}")
        report.append(f"Number of Predictions: {metrics['num_predictions']}")

        if metrics['num_actual_events'] > 0:
            report.append(f"\nActual Event Types: {', '.join(metrics['actual_event_types'])}")
            report.append(f"Predicted Event Types (top 5): {', '.join(metrics['predicted_event_types'])}")

            # Hit@k metrics
            report.append("\n" + "-" * 80)
            report.append("HIT@K METRICS (Does any actual event appear in top-k predictions?)")
            report.append("-" * 80)

            for k in [1, 3, 5]:
                if f'hit@{k}' in metrics:
                    hit_score = metrics[f'hit@{k}']
                    matches = metrics[f'hit@{k}_matches']
                    status = "[HIT]" if hit_score == 1.0 else "[MISS]"
                    report.append(f"Hit@{k}: {hit_score:.2f} {status}")
                    if matches:
                        report.append(f"  -> Matched types: {', '.join(matches)}")

            # MRR
            report.append("\n" + "-" * 80)
            report.append("MEAN RECIPROCAL RANK (MRR)")
            report.append("-" * 80)
            mrr = metrics['mrr']
            rank = metrics['first_match_rank']

            if rank:
                report.append(f"MRR: {mrr:.4f} (first match at rank {rank})")
            else:
                report.append(f"MRR: {mrr:.4f} (no match found)")

            # Type accuracy
            report.append("\n" + "-" * 80)
            report.append("TYPE ACCURACY (Top prediction exact match)")
            report.append("-" * 80)
            acc = metrics['type_accuracy']
            status = "[CORRECT]" if acc == 1.0 else "[INCORRECT]"
            report.append(f"Accuracy: {acc:.2f} {status}")
            if metrics['exact_match']:
                report.append(f"  -> Matched: {', '.join(metrics['exact_match'])}")
        else:
            report.append("\n[WARNING] No actual events found in the prediction horizon.")
            report.append("  Cannot calculate metrics without ground truth.")

        report.append("=" * 80)

        return "\n".join(report)


def parse_predictions_from_result(result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse predictions from orchestrator result.

    Args:
        result_dict: Dictionary result from orchestrator (when output_format='dict')

    Returns:
        List of prediction dictionaries
    """
    if 'ranked_predictions' in result_dict:
        predictions = []
        for pred in result_dict['ranked_predictions']:
            predictions.append({
                'event_type': pred.get('event_type', ''),
                'confidence': pred.get('confidence', 0.0),
                'actors': pred.get('actors', []),
                'location': pred.get('location', ''),
                'date': pred.get('predicted_date', ''),
                'intensity': pred.get('intensity', 0.0)
            })
        return predictions

    return []

if __name__ == "__main__":
    
    # Simulate data loading (similar to the previous step's output)
    event_data_df = pd.read_csv(r"C:\Users\Mitsy\itcs6190\TMS\data\IND_test.csv")

    # FIX: Convert the DataFrame into the required list of CausalMemoryEvent objects
    event_list = convert_csv_to_events(event_data_df)
    print(f"Converted {len(event_list)} rows to CausalMemoryEvents.")

    # Build the TKG
    TKG = build_temporal_knowledge_graph(
        event_list, # Pass the converted list, not the DataFrame
        time_window_days=30,
        max_lookback_events=10
    )
    test_index = TKGSearchIndex(event_list)

    nx.write_gml(TKG, "tkg_test_memory_loc2.gml")
    print("\nGraph saved to tkg_test_memory_loc2.gml (Persistence Checkpoint).")

    #create orchestrator
    orchestrator = create_orchestrator(tkg=TKG, event_list=event_list)
