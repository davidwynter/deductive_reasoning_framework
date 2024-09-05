
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from deductive_ai.engine.confidence_model import ConfidenceAdjuster

def load_training_data(csv_file):
    import csv
    training_data = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            subject, predicate, object_, confidence = row
            confidence = float(confidence)
            training_data.append(((subject, predicate, object_), confidence))
    return training_data

def test_train_save_load_model():
    # Load training data from CSV
    training_data = load_training_data('training_data.csv')

    # Initialize the ConfidenceAdjuster and train the model
    model = ConfidenceAdjuster(input_size=3)  # Placeholder input_size
    model.train_ml_model(training_data, epochs=50)

    # Save the trained model
    model.save_ml_model('confidence_adjuster.pth')

    # Load the model back
    model.load_ml_model('confidence_adjuster.pth')

    # Test apply_ml_model method
    fact = ('Entity1', 'is_related_to', 'Entity2')
    adjusted_confidence = model.apply_ml_model(fact)
    print(f'Adjusted confidence for {fact}: {adjusted_confidence:.4f}')

def test_combine_confidences():
    # Initialize the ConfidenceAdjuster
    model = ConfidenceAdjuster(input_size=3)  # Placeholder input_size

    deductive_confidence = 0.8
    ml_confidence = 0.9

    combined_confidence = model.combine_confidences(deductive_confidence, ml_confidence)
    print(f'Combined confidence: {combined_confidence:.4f}')

if __name__ == "__main__":
    test_train_save_load_model()
    test_combine_confidences()
