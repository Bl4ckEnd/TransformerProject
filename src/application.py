import streamlit as st
import yaml

# let user decide in streamlit application which parameters to use for training
st.title("Methods in Artificial Intelligence - Transformer Project")
st.markdown("""In this streamlit application you can train a transformer model to classify
movie reviews into positive or negative. Notice that inside the parameters, you can find
'Number of Samples'. This refers to the size of the whole dataset that will be afterwards
split into train (80%) and test (20%).""")
st.markdown("**Step 1**: Choose hyperparameters and press 'Confirm parameters'.")
h = st.slider("h (Number of Attention-Heads)", 1, 8, 2)
N = st.slider("N (Number of Attention-Layers)", 1, 6, 2)
epochs = st.slider("Epochs", 1, 10, 5)
n_samples = st.slider("Number of Samples", 5000, 50000, 10000, step=5000)

# confirmed variables
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

if "training_finished" not in st.session_state:
    st.session_state.training_finished = False

if "testing_finished" not in st.session_state:
    st.session_state.testing_finished = False

st.markdown("*Hint: larger parameters lead to longer training times.*")

if st.button("Confirm parameters"):
    # change parameters in params.yaml
    with open("params.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["model"]["h"] = h
    config["model"]["N"] = N

    # calculate amount of data to use (between 0 and 1)
    amount_of_data = n_samples / 50000
    config["training"]["amount_of_data"] = amount_of_data
    config["training"]["epochs"] = epochs

    # write new parameters to params.yaml
    with open("params.yaml", "w") as f:
        yaml.dump(config, f)

    st.session_state.confirmed = True

if st.session_state.confirmed:
    st.write("Parameters confirmed")
    st.markdown("*Hint 1: On Mac M1 chips, first training loss might be infinite.*")
    st.markdown("*Hint 2: If training loss doesn't improve, the model is stuck. Try to change parameters and re-run.*")
    st.markdown("**Step 2**: Start training.")

if st.button("Start training") and st.session_state.confirmed:
    # start training
    from training import train

    st.write("Training started")
    st.session_state.model = train()
    st.write("Training finished")
    st.session_state.training_finished = True


if st.session_state.training_finished:
    st.markdown("**Step 3**: Press 'Start test' to calculate test error")
    if st.button("Start test"):
        from testing import test

        st.write("Test started")
        test(st.session_state.model)
        st.write("Test completed")
        st.session_state.testing_finished = True

if st.session_state.testing_finished:
    st.write("Session finished.")