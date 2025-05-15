"""
Confidence Model for the Deductive Reasoning Framework.
This module provides a ConfidenceAdjuster class that can be used to adjust confidence scores
for inferences based on various factors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

class ConfidenceModel(nn.Module):
    """
    Neural network model for adjusting confidence scores.
    """
    def __init__(self, input_size, hidden_size=64):
        super(ConfidenceModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

class ConfidenceAdjuster:
    """
    Class for adjusting confidence scores based on a trained model.
    """
    def __init__(self, input_size=10, hidden_size=64):
        self.model = ConfidenceModel(input_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.input_size = input_size
        
    def train(self, training_data, epochs=100, learning_rate=0.001):
        """
        Train the confidence model on the provided training data.
        
        Args:
            training_data: List of tuples ((subject, predicate, object), confidence)
            epochs: Number of training epochs
            learning_rate: Learning rate for the optimizer
        """
        # Update optimizer with new learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert training data to tensors
        X = []
        y = []
        for (s, p, o), confidence in training_data:
            # Convert triple to feature vector (this is a simplified example)
            # In a real implementation, you would use embeddings or other features
            feature_vector = torch.zeros(self.input_size)
            # Use hash of strings to create feature indices
            s_idx = hash(s) % (self.input_size // 3)
            p_idx = (hash(p) % (self.input_size // 3)) + (self.input_size // 3)
            o_idx = (hash(o) % (self.input_size // 3)) + (2 * self.input_size // 3)
            
            feature_vector[s_idx] = 1.0
            feature_vector[p_idx] = 1.0
            feature_vector[o_idx] = 1.0
            
            X.append(feature_vector)
            y.append(torch.tensor([confidence], dtype=torch.float32))
        
        # Train the model
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(X[i])
                loss = self.criterion(output, y[i])
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X):.4f}')
    
    def predict(self, triple):
        """
        Predict confidence score for a given triple.
        
        Args:
            triple: Tuple (subject, predicate, object)
            
        Returns:
            Confidence score between 0 and 1
        """
        self.model.eval()
        
        # Convert triple to feature vector
        s, p, o = triple
        feature_vector = torch.zeros(self.input_size)
        
        # Use hash of strings to create feature indices
        s_idx = hash(s) % (self.input_size // 3)
        p_idx = (hash(p) % (self.input_size // 3)) + (self.input_size // 3)
        o_idx = (hash(o) % (self.input_size // 3)) + (2 * self.input_size // 3)
        
        feature_vector[s_idx] = 1.0
        feature_vector[p_idx] = 1.0
        feature_vector[o_idx] = 1.0
        
        # Make prediction
        with torch.no_grad():
            confidence = self.model(feature_vector).item()
        
        return confidence
    
    def save(self, path):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state and metadata
        model_data = {
            'state_dict': self.model.state_dict(),
            'input_size': self.input_size
        }
        torch.save(model_data, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            model_data = torch.load(path)
            self.input_size = model_data['input_size']
            self.model = ConfidenceModel(self.input_size)
            self.model.load_state_dict(model_data['state_dict'])
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            print(f"Model loaded from {path}")
        else:
            print(f"Model file {path} not found")