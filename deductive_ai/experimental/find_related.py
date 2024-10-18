# pip install rdflib
# pip install nltk scikit-learn
# pip install transformers datasets

from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

"""
Could be useful - https://www.analyticsvidhya.com/blog/2020/03/6-pretrained-models-text-classification/
We only use the Bert model here
"""

def tag_entity_only():
    g = Graph()

    # Define some namespaces
    EX = Namespace("http://example.org/")
    FASHION = Namespace("http://example.org/fashion/")

    g.bind("ex", EX)
    g.bind("fashion", FASHION)

    # Add some triples to the graph
    michelle = URIRef("http://example.org/people/MichelleObama")
    karl = URIRef("http://example.org/people/KarlLagerfeld")
    red_dress = URIRef("http://example.org/items/RedDress")

    g.add((michelle, RDF.type, EX.Person))
    g.add((michelle, EX.wears, red_dress))
    g.add((red_dress, EX.designedBy, karl))
    g.add((red_dress, RDF.type, FASHION.Clothing))
    g.add((karl, RDF.type, EX.Person))
    g.add((karl, EX.occupation, FASHION.Designer))

    query = """
    PREFIX ex: <http://example.org/>
    PREFIX fashion: <http://example.org/fashion/>

    SELECT ?s ?p ?o
    WHERE {
        ?s ?p ?o .
        ?s ?p ?o .
        FILTER (?s = <http://example.org/people/MichelleObama> && ?o IN (fashion:Clothing, fashion:Designer))
    }
    """

    qres = g.query(query)

    for row in qres:
        print(f"{row.s} {row.p} {row.o}")

def labelled_dataset():
    # Sample data
    sentences = [
        "Michelle Obama wears a red dress designed by Karl Lagerfeld",
        "Lionel Messi scores a hat-trick in the Champions League",
        "The stock market crashed due to economic instability",
        "The new fashion line by Dior is stunning"
    ]
    labels = ["fashion", "sports", "economics", "fashion"]

    # Preprocess the text
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(tokens)

    processed_sentences = [preprocess(sentence) for sentence in sentences]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(processed_sentences, labels, test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF and SVM
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Classify a new sentence
    new_sentence = "Michelle Obama wears a red dress designed by Karl Lagerfeld"
    processed_new_sentence = preprocess(new_sentence)
    prediction = model.predict([processed_new_sentence])
    print(f"Prediction: {prediction[0]}")


"""
This example demonstrates how to fine-tune BERT on the AG News dataset, 
which is a common text classification dataset.

With an unlabelled dataset topic modeling (e.g., LDA) 
can help identify the main topics in your text, 
which can then be used to create labels for training a classifier.
"""
def train_bert():
    # Load the dataset
    dataset = load_dataset('ag_news')

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    # Train the model
    trainer.train()

def create_tag_query_kg():
    # Sample data for training the model, tagged with domain specific tags
    data = {
        'text': [
            "Michelle Obama wears a red dress designed by Karl Lagerfeld",
            "Lionel Messi scores a hat-trick in the Champions League",
            "The stock market crashed due to economic instability",
            "The new fashion line by Dior is stunning"
        ],
        'label': [0, 1, 2, 0]  # 0: fashion, 1: sports, 2: economics
    }

    dataset = Dataset.from_dict(data)

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train the model
    trainer.train()

    subject_domain_to_namespace = {
        "fashion": "http://example.org/fashion/",
        "sports": "http://example.org/sports/",
        "economics": "http://example.org/economics/"
    }

    def classify_sentence(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        return predictions.item()

    # Example sentence
    sentence = "Michelle Obama wears a red dress designed by Karl Lagerfeld"
    subject_domain_index = classify_sentence(sentence)
    subject_domains = ["fashion", "sports", "economics"]
    subject_domain = subject_domains[subject_domain_index]
    print(f"Subject Domain: {subject_domain}")

    g = Graph()

    # Define namespaces
    subject_domain_to_namespace = {
        "fashion": "http://example.org/fashion/",
        "sports": "http://example.org/sports/",
        "economics": "http://example.org/economics/"
    }

    EX = Namespace("http://example.org/")
    namespace = Namespace(subject_domain_to_namespace[subject_domain])

    g.bind("ex", EX)
    g.bind(subject_domain, namespace)

    # Add triples to the graph
    michelle = URIRef("http://example.org/people/MichelleObama")
    karl = URIRef("http://example.org/people/KarlLagerfeld")
    red_dress = URIRef("http://example.org/items/RedDress")

    g.add((michelle, RDF.type, EX.Person))
    g.add((michelle, EX.wears, red_dress))
    g.add((red_dress, EX.designedBy, karl))
    g.add((red_dress, RDF.type, namespace.Clothing))
    g.add((karl, RDF.type, EX.Person))
    g.add((karl, EX.occupation, namespace.Designer))

    # Define a SPARQL query to filter triples based on the subject domain
    query = f"""
    PREFIX ex: <http://example.org/>
    PREFIX {subject_domain}: <{subject_domain_to_namespace[subject_domain]}>

    SELECT ?s ?p ?o
    WHERE {{
        ?s ?p ?o .
        FILTER (?o IN ({subject_domain}:Clothing, {subject_domain}:Designer))
    }}
    """

    qres = g.query(query)

    for row in qres:
        print(f"{row.s} {row.p} {row.o}")
