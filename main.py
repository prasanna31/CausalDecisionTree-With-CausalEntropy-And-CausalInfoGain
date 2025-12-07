from info_theory_cdt import InformationTheoryCDT
from pamch_cdt import MantelHaenszelCDT
from standard_dt import StandardDecisionTree
from data_preprocessing import Preprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import argparse


warnings.filterwarnings("ignore")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Causal Decision Tree Runner")
    # parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file')
    # parser.add_argument('--target', type=str, required=True, help='Target column name')
    # parser.add_argument('--method', type=str, choices=['IT', 'MH','STD'], required=True, help='Causal Decision Tree method to use: IT for Information Theory, MH for Mantel-Haenszel')
    # args = parser.parse_args()
    # dp = Preprocessor(args.dataset, args.target)
    # df = dp.load_and_preprocess()
    # X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[args.target]), df[args.target], test_size=0.2, random_state=42)
    # train_data = pd.concat([X_train, y_train], axis=1)
    # test_data = pd.concat([X_test, y_test], axis=1)

    # if args.method == 'IT':
    #     print("Using Information Theory CDT...")
    #     model = InformationTheoryCDT(max_height=10, min_samples=20, alpha=0.05, mode='PS', sps_groups=5, min_ig=1e-4)
    # elif args.method == 'MH':
    #     print("Using Mantel-Haenszel CDT...")
    #     model = MantelHaenszelCDT(max_height=10, min_samples=20, alpha=0.05, mode='PS', sps_groups=5)
    # elif args.method == 'STD':
    #     print("Using Standard Decision Tree...")
    #     model = StandardDecisionTree(criterion='entropy', max_depth=10)
    # else:
    #     raise ValueError("Invalid method. Choose either 'IT' or 'MH'.")

    # print("Training the model...")
    # if args.method in ['IT', 'MH']:
    #     model.fit(train_data, target=args.target, predictors=X_train.columns.tolist())
    #     predictions = model.predict(test_data)
    # else:
    #     model.train(X_train, y_train)
    #     predictions = model.predict(X_test)
    # accuracy = (predictions == y_test).mean()
    # print(f"Model Accuracy: {accuracy * 100:.2f}%")

    from info_theory_cdt import InformationTheoryCDT
from pamch_cdt import MantelHaenszelCDT
from standard_dt import StandardDecisionTree
from data_preprocessing import Preprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Define all datasets and their specific target columns here
DATASETS_CONFIG = [
    {"name": "Adult", "path": "./datasets/adult.data", "target": ">50K"},
    {"name": "German Credit", "path": "./datasets/german.data", "target": "bad_credit"},
    {"name": "Car Evaluation", "path": "./datasets/car.data", "target": "acceptable"},
    {"name": "Hypothyroid", "path": "./datasets/hypothyroid.data", "target": "hypothyroid"},
    {"name": "House Votes", "path": "./datasets/house-votes-84.data", "target": "republican"},
    {"name": "Breast Cancer", "path": "./datasets/breast-cancer-wisconsin.data", "target": "malignant"},
    {"name": "Mushroom", "path": "./datasets/agaricus-lepiota.data", "target": "poisonous"},
    {"name": "Chess (Kr-vs-Kp)", "path": "./datasets/kr-vs-kp.data", "target": "won"},
]

METHODS = ['IT', 'MH', 'STD']

def run_experiment():
    results = []
    
    print(f"{'Dataset':<20} | {'Method':<6} | {'Status':<10}")
    print("-" * 45)

    for ds in DATASETS_CONFIG:
        try:
            # 1. Load and Preprocess Data (Once per dataset)
            dp = Preprocessor(ds['path'], ds['target'])
            df = dp.load_and_preprocess()
            
            # 2. Split Data
            X = df.drop(columns=[ds['target']])
            y = df[ds['target']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Prepare data formats for different models
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            # 3. Loop through all methods
            for method in METHODS:
                try:
                    # Initialize Model
                    if method == 'IT':
                        model = InformationTheoryCDT(max_height=10, min_samples=20, alpha=0.05, mode='PS', sps_groups=5, min_ig=1e-4)
                    elif method == 'MH':
                        model = MantelHaenszelCDT(max_height=10, min_samples=20, alpha=0.05, mode='PS', sps_groups=5)
                    elif method == 'STD':
                        model = StandardDecisionTree(criterion='entropy', max_depth=10)

                    # Train and Predict
                    if method in ['IT', 'MH']:
                        model.fit(train_data, target=ds['target'], predictors=X_train.columns.tolist())
                        predictions = model.predict(test_data)
                    else:
                        model.train(X_train, y_train)
                        predictions = model.predict(X_test)

                    # Calculate Accuracy
                    accuracy = (predictions == y_test).mean()
                    
                    # Store Result
                    results.append({
                        "Dataset": ds['name'],
                        "Method": method,
                        "Accuracy": accuracy
                    })
                    print(f"{ds['name']:<20} | {method:<6} | Done ({accuracy:.2%})")

                except Exception as e:
                    print(f"{ds['name']:<20} | {method:<6} | FAILED: {str(e)}")
                    results.append({"Dataset": ds['name'], "Method": method, "Accuracy": None})

        except Exception as e:
            print(f"Could not load {ds['name']}: {e}")

    # --- FINAL OUTPUT ---
    print("\n" + "="*50)
    print("FINAL COMPARISON TABLE")
    print("="*50)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Format accuracy as percentage string for display
        results_df['Accuracy'] = results_df['Accuracy'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "Error")
        
        # Pivot table: Rows = Datasets, Columns = Methods
        pivot_table = results_df.pivot(index="Dataset", columns="Method", values="Accuracy")
        
        print(pivot_table)
        
        # Optional: Save to CSV
        pivot_table.to_csv("experiment_results.csv")
        print("\n[Saved results to 'experiment_results.csv']")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_experiment()