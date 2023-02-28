import streamlit as st
import yaml

# let user decide in streamlit application which parameters to use for training
st.text("Choose hyperparameters")
h = st.slider("h", 1, 8, 2)
N = st.slider("N", 1, 6, 2)
epochs = st.slider("epochs", 1, 10, 5)
n_samples = st.slider("n_samples", 5000, 50000, 10000, step=5000)

# confirmed variables
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

if "training_finished" not in st.session_state:
    st.session_state.training_finished = False

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

if st.button("Start training") and st.session_state.confirmed:
    # start training
    from training import train

    st.write("Training started")
    st.session_state.model = train()
    st.write("Training finished")
    st.session_state.training_finished = True

if st.button("Start test") and st.session_state.training_finished:
    from testing import test

    st.write("Test started")
    test(st.session_state.model)
    st.write("Test completed")
