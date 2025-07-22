# User-transaction-"""
DeFi Credit Scoring Model for Aave V2 Protocol
Assigns credit scores (0-1000) to wallets based on transaction behavior
"""

# Check for required packages
required_packages = ['pandas', 'numpy', 'sklearn']
missing_packages = []

for package in required_packages:
    try:
        if package == 'sklearn':
            import sklearn
        else:
            __import__(package)
    except ImportError:
        if package == 'sklearn':
            missing_packages.append('scikit-learn')
        else:
            missing_packages.append(package)

if missing_packages:
    print("ERROR: Missing required packages!")
    print("Please install them by running:")
    print(f"pip install {' '.join(missing_packages)}")
    exit(1)

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DeFiCreditScorer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = []

    def load_data(self, json_file_path):
        """Load and clean transaction data from JSON"""
        print("Loading transaction data...")
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Could not find file '{json_file_path}'")
            return None
        except json.JSONDecodeError:
            print(f"ERROR: Invalid JSON file '{json_file_path}'")
            return None

        # Flatten nested fields
        df = pd.json_normalize(data)

        # Rename to expected format
        df.rename(columns={
            'userWallet': 'wallet',
            'actionData.amount': 'amount'
        }, inplace=True)

        # Convert types
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce') / 1e6

        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            except:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        print(f"Loaded {len(df)} transactions for {df['wallet'].nunique()} unique wallets")
        return df

    def engineer_features(self, df):
        """Engineer features from transaction data"""
        print("Engineering features...")
        wallet_features = []

        for wallet, wallet_txns in df.groupby('wallet'):
            features = {'wallet': wallet}
            features['total_transactions'] = len(wallet_txns)
            features['unique_actions'] = wallet_txns['action'].nunique()
            if 'timestamp' in wallet_txns.columns and not wallet_txns['timestamp'].isna().all():
                features['days_active'] = max((wallet_txns['timestamp'].max() - wallet_txns['timestamp'].min()).days + 1, 1)
                features['avg_txns_per_day'] = features['total_transactions'] / features['days_active']
            else:
                features['days_active'] = 1
                features['avg_txns_per_day'] = features['total_transactions']

            action_counts = wallet_txns['action'].value_counts()
            total_actions = len(wallet_txns)
            features['deposit_ratio'] = action_counts.get('deposit', 0) / total_actions
            features['borrow_ratio'] = action_counts.get('borrow', 0) / total_actions
            features['repay_ratio'] = action_counts.get('repay', 0) / total_actions
            features['redeem_ratio'] = action_counts.get('redeemunderlying', 0) / total_actions
            features['liquidation_ratio'] = action_counts.get('liquidationcall', 0) / total_actions

            deposits = wallet_txns[wallet_txns['action'] == 'deposit']['amount'].sum()
            borrows = wallet_txns[wallet_txns['action'] == 'borrow']['amount'].sum()
            repays = wallet_txns[wallet_txns['action'] == 'repay']['amount'].sum()
            features['total_deposit_volume'] = deposits
            features['total_borrow_volume'] = borrows
            features['total_repay_volume'] = repays
            features['net_position'] = deposits - borrows + repays
            features['borrow_to_deposit_ratio'] = borrows / max(deposits, 1)
            features['repay_to_borrow_ratio'] = repays / max(borrows, 1)

            features['was_liquidated'] = int('liquidationcall' in action_counts.index)
            features['liquidation_frequency'] = action_counts.get('liquidationcall', 0)
            features['avg_amount_per_txn'] = wallet_txns['amount'].mean()
            features['amount_volatility'] = wallet_txns['amount'].std() / max(wallet_txns['amount'].mean(), 1)

            if 'timestamp' in wallet_txns.columns and not wallet_txns['timestamp'].isna().all():
                tx_sorted = wallet_txns.sort_values('timestamp')
                time_diffs = tx_sorted['timestamp'].diff().dt.total_seconds() / 3600
                features['avg_time_between_txns'] = time_diffs.mean() if not time_diffs.isna().all() else 24
                features['txn_frequency_consistency'] = 1 / max(time_diffs.std() / max(time_diffs.mean(), 1), 1) if not time_diffs.isna().all() else 1
            else:
                features['avg_time_between_txns'] = 24
                features['txn_frequency_consistency'] = 1

            features['same_amount_txns'] = (wallet_txns['amount'].value_counts().max() / len(wallet_txns))
            features['round_amount_ratio'] = sum(wallet_txns['amount'] % 1 == 0) / len(wallet_txns)

            wallet_features.append(features)

        features_df = pd.DataFrame(wallet_features).fillna(0)
        print(f"Engineered {len(features_df.columns)-1} features for {len(features_df)} wallets")
        return features_df

    def create_credit_labels(self, features_df):
        print("Creating credit score labels...")
        scores = np.zeros(len(features_df))

        for i, row in features_df.iterrows():
            score = 500
            score += min(row['repay_to_borrow_ratio'] * 200, 200)
            score += min(row['days_active'] * 2, 100)
            score += min(row['total_transactions'] * 5, 100)
            score += min(row['unique_actions'] * 30, 120)
            score -= row['liquidation_frequency'] * 150
            score -= row['was_liquidated'] * 100
            score -= min(row['same_amount_txns'] * 200, 200)
            score -= max(0, (row['borrow_to_deposit_ratio'] - 0.8) * 300)

            if row['txn_frequency_consistency'] > 0.5:
                score += 50
            if 0.2 <= row['repay_ratio'] <= 0.8:
                score += 50

            scores[i] = max(0, min(1000, score))
        return scores

    def train_model(self, features_df, scores):
        print("Training credit scoring model...")
        X = features_df.drop('wallet', axis=1)
        self.feature_columns = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        print(f"Model R² Score - Train: {self.model.score(X_train_scaled, y_train):.3f}, Test: {self.model.score(X_test_scaled, y_test):.3f}")
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
        return importance

    def predict_scores(self, features_df):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        X = features_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, 1000)

    def generate_wallet_scores(self, json_file_path, output_file=None):
        print("=== DeFi Credit Scoring Pipeline ===")
        df = self.load_data(json_file_path)
        if df is None:
            return None, None

        features_df = self.engineer_features(df)
        training_scores = self.create_credit_labels(features_df)
        feature_importance = self.train_model(features_df, training_scores)
        final_scores = self.predict_scores(features_df)

        results = pd.DataFrame({
            'wallet': features_df['wallet'],
            'credit_score': final_scores.round().astype(int),
            'total_transactions': features_df['total_transactions'],
            'days_active': features_df['days_active'],
            'repay_ratio': features_df['repay_ratio'].round(3),
            'was_liquidated': features_df['was_liquidated'].astype(int)
        }).sort_values('credit_score', ascending=False)

        print(f"\n=== Credit Score Distribution ===")
        print(f"Mean Score: {results['credit_score'].mean():.1f}")
        print(f"Median Score: {results['credit_score'].median():.1f}")
        print(f"Score Range: {results['credit_score'].min()} - {results['credit_score'].max()}")

        print("\nScore Percentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"{p}th percentile: {results['credit_score'].quantile(p/100):.0f}")

        print("\nTop 10 Highest Scored Wallets:")
        print(results.head(10).to_string(index=False))

        print("\nBottom 10 Lowest Scored Wallets:")
        print(results.tail(10).to_string(index=False))

        if output_file:
            results.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")

        return results, feature_importance


def main():
    scorer = DeFiCreditScorer()
    json_file_path = "user-transactions.json"
    print("Starting DeFi Credit Scoring...")
    print(f"Looking for file: {json_file_path}")
    print("-" * 50)

    try:
        results, feature_importance = scorer.generate_wallet_scores(
            json_file_path,
            output_file="wallet_credit_scores.csv"
        )
        if results is not None:
            print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
