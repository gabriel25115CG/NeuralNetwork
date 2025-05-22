# Simple Neural Network for Real Estate Price Prediction

This project implements a simple neural network from scratch in Python to predict real estate prices based on surface area (in square meters). It includes data normalization, a training routine using gradient descent, loss visualization, and interactive predictions via terminal input.

## Features:

* Custom neural network implementation without external machine learning libraries
* Layer-based architecture with forward propagation
* Gradient descent training on a single neuron
* Data normalization and denormalization
* Interactive predictions from user input
* Visualizations of training loss and prediction results

## Project Structure:

* `main.py`: Entry point with training loop and predictions
* `data/dataset.py`: Dataset and normalization variables
* `neural_network/network.py`: `NeuralNetwork` class
* `neural_network/layer.py`: Layer and neuron definitions
* `utils/visualization.py`: Plotting functions for loss and predictions

## Requirements:

* Python 3.10+
* matplotlib (for plotting)

## Usage:

Run `python main.py` to train the model and make predictions.

After training, enter surface areas in m² to get predicted prices interactively.

## Example Output:

```yaml
Epoch 200/200, Loss: 0.040645, Weight: 0.225282, Bias: 0.465310
Surface: 120.0 m² => Predicted price: 476.50 thousands $
Surface: 180.0 m² => Predicted price: 517.05 thousands $
Surface: 220.0 m² => Predicted price: 544.08 thousands $