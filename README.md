# Deductive Reasoning Framework

## Overview

The Deductive Reasoning Framework is a Python-based application that leverages various machine learning models and RDF (Resource Description Framework) data formats to perform complex reasoning tasks. The project integrates multiple probabilistic models and provides a user-friendly interface via Streamlit, enabling users to manage data, perform inference, and optimize the contribution of different models through hyper-parameter tuning.

## Table of Contents

- [Deductive Reasoning Framework](#deductive-reasoning-framework)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Project Structure](#project-structure)
  - [Modules and Classes](#modules-and-classes)
    - [`engine/` - Core Engine Components](#engine---core-engine-components)
    - [`/` - Streamlit Page Components](#---streamlit-page-components)
    - [`utils/` - Utility Functions](#utils---utility-functions)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [VS Code launch.json entry](#vs-code-launchjson-entry)

## Installation

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Optional: `python-dotenv` if environment variables are needed

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/davidwynter/deductive_reasoning_framework.git
   cd deductive_reasoning_framework
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Run the Streamlit application:**
   ```bash
   poetry run streamlit run src/deductive_ai/app.py
   ```

## Project Structure

```
deductive_reasoning_framework/
│
├── src/
|   ├── deductive_ai/
|       ├── app.py                  # Main Streamlit application
│       │── login.py                # Login page
│       │── data_management.py      # Data & Rule Management page
│       │── inference_validation.py # Inference & Validation page
│       │── hyperparameter_tuning_page.py # Hyper-Parameter Tuning page
│       ├── __init__.py               # Package initializer
│       │
│       ├── engine/                   # Core engine components
│       │   ├── __init__.py           # Engine module initializer
│       │   ├── deductive_engine.py   # DeductiveReasoningEngine class
│       │   ├── hyperparameter_tuning.py # HyperParameterTuning class
│       │   ├── text_to_rdf.py        # TextToRDFConverter class
│       │   └── authentication.py     # Authentication and user management functions
│       │
│       ├── utils/                    # Utility functions
│       │   ├── __init__.py           # Utils module initializer
│       │   ├── file_utils.py         # Utility functions for file handling
│       │   ├── rdf_utils.py          # Utility functions for RDF handling
│       │   └── env_utils.py          # Utility functions for environment variables (if needed)
│
├── .env                          # Environment variables (optional)
├── pyproject.toml                # Poetry configuration file
├── README.md                     # Project documentation
└── users.json                    # User data file (created at runtime)
```

## Modules and Classes

### `engine/` - Core Engine Components

- **`deductive_engine.py`**
  - **DeductiveReasoningEngine**: This is the main engine of the framework. It handles reasoning processes, manages confidence estimation using different ML models, and allows the user to load datasets and rules for inference.

- **`hyperparameter_tuning.py`**
  - **HyperParameterTuning**: This class integrates with the Ray Tune library to optimize the contribution of various ML models. It runs hyper-parameter tuning to find the best combination of model contributions that maximize matched triples during inference.

- **`text_to_rdf.py`**
  - **TextToRDFConverter**: Converts unstructured text into RDF triples using the OpenIE approach. The extracted triples are then serialized into TTL or N3 formats and loaded into the RDF graph.

- **`authentication.py`**
  - Contains functions for user authentication and management. It handles password encryption, role-based access control, and allows "Admin" users to create new users.

### `/` - Streamlit Page Components

- **`login.py`**
  - Implements the login page, which includes password change functionality for users. It also provides the ability for "Admin" users to create new users.

- **`data_management.py`**
  - Implements the Data & Rule Management page. Users can upload files, enter rules, and manage datasets. It supports conversion of unstructured text to RDF and allows for rule entry in N3 or SWRL formats.

- **`inference_validation.py`**
  - Implements the Inference & Validation page. Users can select datasets, run inference using the loaded data, and validate the results against expected triples.

- **`hyperparameter_tuning_page.py`**
  - Implements the Hyper-Parameter Tuning page. This page allows users to configure and run hyper-parameter tuning using Ray Tune to optimize the contribution of different ML models.

### `utils/` - Utility Functions

- **`file_utils.py`**
  - Contains utility functions for handling file uploads, conversion, and storage.

- **`rdf_utils.py`**
  - Contains utility functions for managing RDF data, including loading, parsing, and converting RDF files.

- **`env_utils.py`**
  - Contains utility functions for managing environment variables, if needed (e.g., for API keys, database credentials).

## Usage

1. **Login:**
   - Start the application and login using the default admin credentials (`admin/admin`). Change the password on the first login.
   - Admin users can create new users and manage their roles.

2. **Data Management:**
   - Upload unstructured text files or RDF files (TTL, N3, XML).
   - Enter rules for reasoning and save them in the desired format (N3 or SWRL).
   - Create named datasets by selecting uploaded files and associating them with expected triples.

3. **Inference & Validation:**
   - Select datasets to run inference against.
   - Choose the ML models to include in the inference process and specify their contribution percentages.
   - Validate the results by comparing the matched triples against the expected set.

4. **Hyper-Parameter Tuning:**
   - Select a dataset and configure the percentage ranges for each ML model.
   - Run Ray Tune to optimize the model contributions and maximize the matched triples during inference.

5. **BaseURL Selection:**
   - The base_url parameter represents the base URI (Uniform Resource Identifier) that is used to create fully qualified URIs for subjects, predicates, and objects in the RDF graph. For example, if you pass http://example.org as the base_url, a subject like "Barack_Obama" will be converted into the URI http://example.org/Barack_Obama.
   - How is base_url Usually Chosen?

    - Domain Name:
        Use of an Organizational Domain: Typically, base_url is chosen based on the domain of the organization or project. For example, if the organization is "Example Corp" with the domain "example.com", the base URL might be http://example.com/data.
        Ensures Uniqueness: Using a domain-controlled URL ensures that the URIs are globally unique and can be dereferenced if published on the web.

    - Context-Specific URIs:
        Project or Dataset-Specific: The base_url may include specific paths related to the project or dataset being managed. For example, http://example.com/dataset1 for a specific dataset.
        Versioning: Sometimes, the base URL may include versioning information, e.g., http://example.com/v1/data.

    - Default or Placeholder URIs:
        Development and Testing: During development or testing, it's common to use placeholders like http://example.org or http://localhost.
        Non-Public Data: If the data is not intended to be public, a generic URL like http://example.org might be used.

    - Best Practices for Choosing base_url:

      Stability: Ensure that the base_url is stable and doesn't change frequently, as URIs are meant to be persistent identifiers.
      Consistency: Use a consistent pattern across datasets and projects within the same domain.
      Descriptive: Where possible, include descriptive elements in the base_url to indicate the nature of the data or its source.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to update the documentation and include tests for any new functionality.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This `README.md` provides a comprehensive overview of your Deductive Reasoning Framework, including how each module and class contributes to the functionality of the project. It also includes installation instructions, usage guidelines, and information about the project structure.

##  VS Code launch.json entry
```
        {
            "name": "Python:Streamlit Deductive",
            "type": "debugpy",
            "request": "launch",
            "module": "streamlit",
            "args": [
                "run",
                "${file}",
                "--server.port",
                "3003"
            ]
        }
```
Select app.py file in editor and run the "Python:Streamlit Deductive" debug entry