from deductive_ai.engine.hyperparameter_tuning import HyperParameterTuning
import streamlit as st

# Hyper-Parameter Tuning Page
def hyperparameter_tuning_page():
    st.title("Hyper-Parameter Tuning")

    # Select the dataset
    dataset = st.selectbox("Select Dataset", options=st.session_state.engine.get_datasets())
    
    # Set ranges for each model's contribution percentage
    st.subheader("Set Ranges for Model Contributions")
    bayesian_range = st.multiselect("Bayesian Networks Range", [10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90, 95, 100], default=[10, 50, 100])
    pymc3_range = st.multiselect("Probabilistic Programming Range", [10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90, 95, 100], default=[10, 50, 100])
    pyro_range = st.multiselect("Monte Carlo Methods Range", [10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90, 95, 100], default=[10, 50, 100])
    pyreason_range = st.multiselect("Markov Logic Networks Range", [10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90, 95, 100], default=[10, 50, 100])

    # Start hyperparameter tuning
    if st.button("Run Hyper-Parameter Tuning"):
        ranges = {
            "Bayesian Networks": bayesian_range,
            "Probabilistic Programming": pymc3_range,
            "Monte Carlo Methods": pyro_range,
            "Markov Logic Networks": pyreason_range
        }
        tuning = HyperParameterTuning(st.session_state.engine, dataset, st.session_state.engine.get_expected_triples(dataset))
        best_config, best_result = tuning.run_tuning(ranges)

        st.success(f"Best Configuration: {best_config}")
        st.metric("Matched Triples", f"{best_result['matched']}%")