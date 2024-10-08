# Fraud Transaction Detection

This project implements a Graph Attention Network (GAT) model for detecting fraudulent transactions in a Bitcoin dataset.

## Technical Details

### Data Processing
- The project uses a Bitcoin transaction dataset with three main components:
  - Transaction classes (elliptic_txs_classes.csv)
  - Transaction edges (elliptic_txs_edgelist.csv)
  - Transaction features (elliptic_txs_features.csv)
- Data is preprocessed using pandas and numpy libraries
- Transactions are mapped to numeric indices for efficient processing

### Model Architecture
- Graph Attention Network (GAT) implemented using PyTorch Geometric
- Model structure:
  - 3 GAT convolution layers
  - ReLU activation and dropout between layers
  - Final sigmoid activation for binary classification

### Key Components
1. `GAT_Model`: Custom PyTorch module implementing the GAT architecture
2. `GnnTrainer`: Class for training the model and managing the training process
3. `MetricManager`: Class for calculating and storing various performance metrics

### Training Process
- Uses Adam optimizer with learning rate scheduling
- Binary Cross Entropy Loss as the loss function
- Implements early stopping based on validation performance
- Tracks multiple metrics including accuracy, F1 score, ROC AUC, precision, and recall

### Libraries Used
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-learn (for metric calculations)
- Matplotlib (for visualizations)

### Hardware Requirements
- Supports GPU acceleration if available, falls back to CPU otherwise

## Usage
1. Ensure all required libraries are installed
2. Place the Bitcoin dataset files in the specified directory
3. Run the Jupyter notebook to train the model and evaluate results

## Future Improvements
- Implement cross-validation for more robust evaluation
- Experiment with different GNN architectures (e.g., GraphSAGE, GIN)
- Add more extensive data visualization and analysis
