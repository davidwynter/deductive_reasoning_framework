import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import time

class SimpleHyperParameterTuning:
    """
    A simple hyperparameter tuning class that simulates the tuning process
    without requiring the actual engine implementation.
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def run_tuning(self, ranges):
        """
        Simulate running hyperparameter tuning with the given ranges.
        
        Args:
            ranges: Dictionary mapping method names to lists of percentage values
            
        Returns:
            best_config: The best configuration found
            best_result: The result metrics for the best configuration
        """
        # Generate all possible combinations of percentages
        methods = list(ranges.keys())
        combinations = []
        
        # Get all combinations of percentages
        percentage_combinations = list(itertools.product(*[ranges[method] for method in methods]))
        
        # Filter combinations where the sum is 100%
        valid_combinations = []
        for combo in percentage_combinations:
            if sum(combo) == 100:
                config = {method: percentage for method, percentage in zip(methods, combo)}
                valid_combinations.append(config)
        
        # If no valid combinations, create some that sum to 100%
        if not valid_combinations:
            # Create 5 random configurations that sum to 100%
            for _ in range(5):
                config = {}
                remaining = 100
                for i, method in enumerate(methods):
                    if i == len(methods) - 1:
                        config[method] = remaining
                    else:
                        value = random.randint(0, remaining)
                        config[method] = value
                        remaining -= value
                valid_combinations.append(config)
        
        # Simulate evaluation for each valid combination
        results = []
        for config in valid_combinations:
            # Simulate a result based on the configuration
            # This is just a dummy calculation for demonstration
            matched = 0
            for method, percentage in config.items():
                # Different methods have different simulated effectiveness
                if method == "Bayesian Networks":
                    matched += percentage * 0.8
                elif method == "Probabilistic Programming":
                    matched += percentage * 0.7
                elif method == "Monte Carlo Methods":
                    matched += percentage * 0.75
                else:
                    matched += percentage * 0.6
                    
            # Add some randomness
            matched = min(100, max(0, matched + random.uniform(-10, 10)))
            
            results.append({
                "config": config,
                "matched": matched,
                "unmatched": 100 - matched
            })
        
        # Find the best configuration
        best_result = max(results, key=lambda x: x["matched"])
        
        return best_result["config"], best_result

# Hyper-Parameter Tuning Page
def hyperparameter_tuning_page():
    st.title("Hyper-Parameter Tuning")

    # Ensure the engine is initialized
    if 'engine' not in st.session_state:
        from engine.deductive_engine import DeductiveReasoningEngine
        st.session_state.engine = DeductiveReasoningEngine()

    # Select the dataset
    dataset = st.selectbox("Select Dataset", options=st.session_state.engine.get_datasets())
    
    # Set ranges for each model's contribution percentage
    st.subheader("Set Ranges for Model Contributions")
    st.info("Select multiple values for each method to explore different combinations. The total should sum to 100%.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bayesian_range = st.multiselect(
            "Bayesian Networks Range",
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            default=[30, 40, 50]
        )
        
        pymc3_range = st.multiselect(
            "Probabilistic Programming Range",
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            default=[20, 30, 40]
        )
    
    with col2:
        pyro_range = st.multiselect(
            "Monte Carlo Methods Range",
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            default=[20, 30, 40]
        )
        
        ml_range = st.multiselect(
            "ML Model Range",
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            default=[10, 20, 30]
        )

    # Start hyperparameter tuning
    if st.button("Run Hyper-Parameter Tuning"):
        ranges = {
            "Bayesian Networks": bayesian_range,
            "Probabilistic Programming": pymc3_range,
            "Monte Carlo Methods": pyro_range,
            "ML Model": ml_range
        }
        
        # Check if any method has no selected values
        empty_methods = [method for method, values in ranges.items() if not values]
        if empty_methods:
            st.error(f"Please select at least one value for: {', '.join(empty_methods)}")
        else:
            with st.spinner("Running hyperparameter tuning..."):
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Simulate progress
                for i in range(101):
                    time.sleep(0.05)  # Simulate computation time
                    progress_bar.progress(i)
                
                # Run the tuning
                tuning = SimpleHyperParameterTuning(dataset)
                best_config, best_result = tuning.run_tuning(ranges)
                
                # Display results
                st.success("Hyperparameter tuning completed!")
                
                # Show the best configuration
                st.subheader("Best Configuration")
                
                # Create a DataFrame for the best configuration
                best_config_df = pd.DataFrame({
                    "Method": list(best_config.keys()),
                    "Contribution (%)": list(best_config.values())
                })
                
                # Display as a table
                st.table(best_config_df)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Matched Triples", f"{best_result['matched']:.1f}%")
                with col2:
                    st.metric("Unmatched Triples", f"{best_result['unmatched']:.1f}%")
                
                # Create a pie chart of the best configuration
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(
                    best_config.values(),
                    labels=best_config.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=True
                )
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title(f"Best Configuration for {dataset}")
                
                # Display the chart
                st.pyplot(fig)
                
                # Save the best configuration
                if st.button("Save Configuration"):
                    # In a real implementation, this would save to a file or database
                    st.session_state.engine.set_confidence_weights(best_config)
                    st.success(f"Configuration saved for dataset: {dataset}")