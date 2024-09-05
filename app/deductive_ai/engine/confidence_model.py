import torch
import torch.nn as nn
import torch.optim as optim

class ConfidenceAdjuster(nn.Module):
    def __init__(self, input_size):
        super(ConfidenceAdjuster, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def apply_ml_model(self, fact):
        """
        Apply the machine learning model to adjust the confidence score.
        """
        if self.confidence_adjuster is None:
            raise ValueError("Confidence adjuster model not loaded. Please load or train a model first.")

        # Convert fact to input tensor
        input_tensor = self.fact_to_tensor(fact)

        # Apply the model
        with torch.no_grad():
            adjusted_confidence = self.confidence_adjuster(input_tensor).item()

        return adjusted_confidence

    def fact_to_tensor(self, fact):
        """
        Convert a fact (triple) to a tensor for input to the ML model.
        This is a simple implementation and may need to be adapted based on your specific needs.
        """
        # Example: use one-hot encoding for subject, predicate, and object
        all_entities = list(set([entity for triple in self.inferred_graph for entity in triple]))
        entity_to_index = {entity: i for i, entity in enumerate(all_entities)}

        tensor = torch.zeros(len(all_entities) * 3)
        tensor[entity_to_index[fact[0]]] = 1
        tensor[len(all_entities) + entity_to_index[fact[1]]] = 1
        tensor[2 * len(all_entities) + entity_to_index[fact[2]]] = 1

        return tensor

    def combine_confidences(self, deductive_confidence, ml_confidence):
        """
        Combine confidence scores from deductive reasoning and the ML model.
        """
        if self.integration_method == "combine_confidence":
            return (deductive_confidence + ml_confidence) / 2
        elif self.integration_method == "weighted_combine":
            return 0.7 * ml_confidence + 0.3 * deductive_confidence
        elif self.integration_method == "ml_override":
            return ml_confidence if ml_confidence else deductive_confidence
        else:
            return deductive_confidence

    def load_ml_model(self, path):
        """
        Load a pre-trained ML model for confidence adjustment.
        """
        input_size = len(self.fact_to_tensor(next(iter(self.inferred_graph))))
        self.confidence_adjuster = ConfidenceAdjuster(input_size)
        self.confidence_adjuster.load_state_dict(torch.load(path))
        self.confidence_adjuster.eval()

    def save_ml_model(self, path):
        """
        Save the trained ML model for confidence adjustment.
        """
        if self.confidence_adjuster is None:
            raise ValueError("No ML model to save. Please train a model first.")
        torch.save(self.confidence_adjuster.state_dict(), path)

    def train_ml_model(self, training_data, epochs=100, learning_rate=0.001):
        """
        Train the ML model for confidence adjustment.
        
        :param training_data: List of tuples (fact, true_confidence)
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        """
        if not training_data:
            raise ValueError("No training data provided.")

        # Initialize the model
        input_size = len(self.fact_to_tensor(training_data[0][0]))
        self.confidence_adjuster = ConfidenceAdjuster(input_size)

        # Prepare the training data
        X = torch.stack([self.fact_to_tensor(fact) for fact, _ in training_data])
        y = torch.tensor([confidence for _, confidence in training_data], dtype=torch.float32).unsqueeze(1)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.confidence_adjuster.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = self.confidence_adjuster(X)
            loss = criterion(outputs, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        print("Training completed.")