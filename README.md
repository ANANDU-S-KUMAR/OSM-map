# OSM Walking Loop Generator

A custom OpenAI Gymnasium environment that uses Reinforcement Learning 
to generate closed walkable routes starting and ending at a fixed location.

## Overview
The agent learns to build walking loops of a user-specified duration (5–50 minutes)
by iteratively adding, removing, or swapping waypoints on a real OpenStreetMap 
graph. The environment is built on OSMnx and NetworkX, and trained using PPO 
from Stable-Baselines3.

## Features
- Real walkable graph loaded from OpenStreetMap via OSMnx
- 4 discrete actions: ADD, REMOVE, SWAP, ACCEPT
- 6-dimensional observation space encoding duration accuracy, 
  loop complexity, and route overlap
- Shaped reward function encouraging duration matching and 
  minimal street repetition
- Visual route display at end of each episode
- Learning curve generated after training

## Tech Stack
- Python, Gymnasium, Stable-Baselines3 (PPO)
- OSMnx, NetworkX, Matplotlib

## Usage
# Train
python model.py  # uncomment train() in main

# Test
python model.py  # enter desired walk duration when prompted
