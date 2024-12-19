# def check_package(package_name):
#     try:
#         __import__(package_name)
#         print(f"{package_name} is installed.")
#     except ImportError:
#         print(f"{package_name} is not installed.")


# # Check if lightgbm is installed
# check_package('lightgbm')

# # Check if rdkit is installed
# check_package('rdkit')


import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data: SMILES strings and target values
smiles_data = [
    "CCO", "CCN", "CCC(=O)O", "CCC(=O)N", "CCC(=O)C"
]
target_values = [1.0, 2.0, 3.0, 4.0, 5.0]

# Function to generate molecular descriptors


def generate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        descriptor_names)
    descriptors = calculator.CalcDescriptors(mol)
    return descriptors


# Generate descriptors for all molecules
descriptors = [generate_descriptors(smiles) for smiles in smiles_data]
descriptors = np.array(descriptors)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    descriptors, target_values, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the model
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[
                test_data], early_stopping_rounds=10)

# Make predictions
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# Save the model
bst.save_model('lightgbm_model.txt')
