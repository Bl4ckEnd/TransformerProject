import streamlit as st
import yaml

# let user decide in streamlit application which parameters to use for training
st.text("Choose hyperparameters to train the model ok")
h = st.slider("h", 1, 8, 2)
N = st.slider("N", 1, 6, 2)
n_samples = st.slider("n_samples", 5000, 50000, 10000, step=5000)

# confirmed variable
confirmed = False

if st.button("Confirm parameters"):
    # change parameters in params.yaml
    with open("src/params.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["model"]["h"] = h
    config["model"]["N"] = N

    # calculate amount of data to use (between 0 and 1)
    amount_of_data = n_samples / 50000
    config["training"]["amount_of_data"] = amount_of_data

    # write new parameters to params.yaml
    with open("src/params.yaml", "w") as f:
        yaml.dump(config, f)

    confirmed = True

if confirmed:
    st.write("Parameters confirmed")

if confirmed and st.button("Start training"):
    # start training
    import src.training

    src.training.train()
