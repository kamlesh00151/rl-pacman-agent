import streamlit as st
from PIL import Image

st.title("RL Pacman AI Project")

st.write("This project uses Deep Q Learning to train Pacman agent.")

st.header("Training Graph")

image = Image.open("training_curves.png")
st.image(image, caption="Training Result")

st.header("Project Info")

st.write("""
- Algorithm: Deep Q Network (DQN)
- Objective: Collect food and avoid ghosts
- Technology: Python, PyTorch, Reinforcement Learning
""")