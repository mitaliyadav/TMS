"""
Example usage of the Multi-Agent Event Prediction System

This script demonstrates how to:
1. Load the TKG and event data once
2. Create an orchestrator
3. Run multiple queries without reloading data
"""

from multi_agent_system import load_data, create_orchestrator

# ============================================================================
# Step 1: Load data ONCE (this is the slow part)
# ============================================================================
print("Step 1: Loading TKG and event data (one-time operation)...")

TKG, event_list = load_data(
    csv_path=r"C:\Users\Mitsy\itcs6190\TMS\data\IND_train.csv",
    tkg_path=r"C:\Users\Mitsy\itcs6190\TMS\tkg_memory_loc2.gml"
)

# ============================================================================
# Step 2: Create orchestrator ONCE
# ============================================================================
print("\nStep 2: Creating orchestrator...")
orchestrator = create_orchestrator(tkg=TKG, event_list=event_list)

# ============================================================================
# Step 3: Run MULTIPLE queries (fast, no reloading)
# ============================================================================
print("\nStep 3: Running multiple queries...\n")

# Example 1: Historical event-based query
print("=" * 80)
print("EXAMPLE 1: Historical Event-Based Query (verbose mode)")
print("=" * 80)

result1 = orchestrator.orchestrate(
    user_query="The Supreme Court accused the Government of Uttar Pradesh regarding healthcare-related issues on May 6, 2022. What were the next set of steps that happened after 2 months?",
    prediction_horizon_days=60,  # 2 months
    output_format="text",
    verbose=True  # Show individual policy predictions
)
print(result1)

print("\n" + "=" * 80)
print("EXAMPLE 2: Historical Event-Based Query (verbose mode)")
print("=" * 80)

result2 = orchestrator.orchestrate(
    user_query="Rahul Gandhi accused Narendra Modi on June 26, 2022. What political developments followed in the next 2 months?",
    prediction_horizon_days=60,  # 2 months
    output_format="text",
    verbose=True  # Show individual policy predictions
)
print(result2)

print("\n" + "=" * 80)
print("EXAMPLE 3: Historical Event-Based Query (verbose mode)")
print("=" * 80)

result3 = orchestrator.orchestrate(
    user_query="Siddaramaiah has accused Bharatiya Janata Party of failing to provide timely and adequate flood relief on September 10, 2022. What actions has Bharatiya Janata Party taken 1 month after this?",
    prediction_horizon_days=30,  # 1 months
    output_format="text",
    verbose=True  # Show individual policy predictions
)
print(result3)
"""
# Example 2: Concise prediction
print("\n\n" + "=" * 80)
print("EXAMPLE 2: Concise mode (shows only final prediction)")
print("=" * 80)

result2 = orchestrator.orchestrate(
    user_query="Will there be political protests in Mumbai next month?",
    prediction_horizon_days=30,
    output_format="text",
    verbose=False  # Show only final prediction
)
print(result2)

# Example 3: Short-term prediction
print("\n\n" + "=" * 80)
print("EXAMPLE 3: Short-term prediction (7 days)")
print("=" * 80)

result3 = orchestrator.orchestrate(
    user_query="What will the BJP do next week?",
    prediction_horizon_days=7,
    output_format="text",
    verbose=False
)
print(result3)
"""

"""
# Example 4: JSON output for API integration
print("\n\n" + "=" * 80)
print("EXAMPLE 4: JSON output format")
print("=" * 80)

result4 = orchestrator.orchestrate(
    user_query="What events will occur in West Bengal in the next 2 weeks?",
    prediction_horizon_days=14,
    output_format="json",  # JSON format for APIs
    verbose=False
)
print(result4)

print("\n" + "=" * 80)
print("All queries completed successfully!")
print("=" * 80)
"""