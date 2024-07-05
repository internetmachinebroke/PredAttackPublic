import requests
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Constants
NEWS_API_KEY = 'NO API KEY FOR YOU'  # 

class CyberAttackPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.label_encoder = LabelEncoder()
        self.stop_words = set(stopwords.words('english'))
        self.industry_mapping = {
            'Healthcare': 'Health Care and Social Assistance',
            'Education': 'Educational Services',
            'Technology': 'Information',
            'Finance': 'Finance and Insurance',
            'Government': 'Public Administration',
            'Retail': 'Retail Trade',
            'Manufacturing': 'Manufacturing',
            'Agriculture': 'Agriculture, Forestry, Fishing and Hunting',
            'Entertainment': 'Arts, Entertainment, and Recreation',
            'Transportation': 'Transportation and Warehousing',
            'Food Service': 'Accommodation and Food Services',
            'Real Estate': 'Real Estate and Rental and Leasing',
            'Mining': 'Mining, Quarrying, and Oil and Gas Extraction',
            'Wholesale': 'Wholesale Trade',
            'Construction': 'Construction',
            'Utilities': 'Utilities',
            'Professional Services': 'Professional, Scientific, and Technical Services',
            'Administrative Services': 'Administrative and Support and Waste Management and Remediation Services',
            'Other Services': 'Other Services (except Public Administration)',
            'Management': 'Management of Companies and Enterprises'
        }

    def fetch_news(self, query='cyberattack', days=30):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f'https://newsapi.org/v2/everything?q={query}&from={start_date.strftime("%Y-%m-%d")}&to={end_date.strftime("%Y-%m-%d")}&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        articles = response.json().get('articles', [])
        news_data = [{'title': article['title'], 'summary': article['description'], 'published_at': article['publishedAt']} for article in articles]
        return pd.DataFrame(news_data)

    def fetch_exploits(self, days=30):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = 'https://www.exploit-db.com/exploits/'
        
        exploits_data = []
        page = 1
        
        while True:
            response = requests.get(f"{url}?page={page}")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = soup.find('table', {'id': 'exploits-table'})
            if not table:
                break
            
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cols = row.find_all('td')
                date = datetime.strptime(cols[1].text.strip(), '%Y-%m-%d')
                
                if date < start_date:
                    return pd.DataFrame(exploits_data)
                
                exploits_data.append({
                    'id': cols[0].text.strip(),
                    'date': date,
                    'title': cols[2].text.strip(),
                    'type': cols[3].text.strip(),
                    'platform': cols[4].text.strip(),
                })
            
            page += 1
        
        return pd.DataFrame(exploits_data)

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        tokens = nltk.word_tokenize(str(text).lower())
        return ' '.join([token for token in tokens if token.isalnum() and token not in self.stop_words])

    def extract_date_features(self, df):
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['day_of_week'] = df['published_at'].dt.dayofweek
        df['month'] = df['published_at'].dt.month
        return df

    def load_historical_data(self):
        # Read data from Excel file
        filename1 = r'C:\Users\gauta\OneDrive\Desktop\Capstone Project (AI)\Prediction Model\umspp-export-2024-06-25.xlsx'
        excel_data = pd.read_excel(filename1)
        
        # Process the data
        processed_data = self._process_excel_data(excel_data)
        
        print(f"Loaded {len(processed_data)} historical incidents.")
        return processed_data

    def _process_excel_data(self, df):
        # Convert event_date to datetime
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'event_date': 'published_at',
            'description': 'text',
            'industry': 'industry'
        })
        
        # Add 'target' column (assuming all entries in this dataset are confirmed incidents)
        df['target'] = 1
        
        # Combine description and additional info
        df['text'] = df['text']
        
        return df[['text', 'published_at', 'target', 'industry']]

    def extract_industry(self, description):
        industries = ['Healthcare', 'Finance', 'Technology', 'Government', 'Education', 'Retail', 'Energy', 'Transportation']
        for industry in industries:
            if industry.lower() in description.lower():
                return industry
        return 'Other'

    def prepare_data(self):
        news_data = self.fetch_news()
        historical_data = self.load_historical_data()
        exploits_data = self.fetch_exploits()

        # Handle potential NaN values
        news_data['title'] = news_data['title'].fillna('')
        news_data['summary'] = news_data['summary'].fillna('')
        news_data['text'] = (news_data['title'] + ' ' + news_data['summary']).apply(self.preprocess_text)

        historical_data['text'] = historical_data['text'].apply(self.preprocess_text)

        news_data = self.extract_date_features(news_data)
        historical_data = self.extract_date_features(historical_data)

        # Combine news and historical data
        combined_data = pd.concat([
            news_data[['text', 'day_of_week', 'month']],
            historical_data[['text', 'day_of_week', 'month', 'target', 'industry']]
        ], ignore_index=True)

        # Add exploit features
        if not exploits_data.empty and 'date' in exploits_data.columns:
            combined_data['exploit_count'] = combined_data['published_at'].apply(
                lambda date: exploits_data[exploits_data['date'].dt.date == date.date()].shape[0]
            )
            combined_data['remote_exploit_count'] = combined_data['published_at'].apply(
                lambda date: exploits_data[(exploits_data['date'].dt.date == date.date()) & 
                                           (exploits_data['type'].str.contains('remote', case=False))].shape[0]
            )
        else:
            print("Warning: 'date' column not found in exploits_data or exploits_data is empty. Setting exploit counts to 0.")
            combined_data['exploit_count'] = 0
            combined_data['remote_exploit_count'] = 0

        # Ensure 'Undetermined' category exists
        if 'Undetermined' not in combined_data['industry'].unique():
            combined_data = combined_data.append({'industry': 'Undetermined'}, ignore_index=True)

        # Encode industry
        combined_data['industry'] = self.label_encoder.fit_transform(combined_data['industry'].fillna('Undetermined'))

        print("Available industry categories:", self.label_encoder.classes_)

        return combined_data

    def train_and_evaluate(self):
        data = self.prepare_data()
        
        X_text = self.vectorizer.fit_transform(data['text'])
        X_features = data[['day_of_week', 'month', 'industry', 'exploit_count', 'remote_exploit_count']].values
        X = np.hstack((X_text.toarray(), X_features))
        y = data['target'].fillna(0).astype(int)

        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        print("Validation Set Performance:")
        print(classification_report(y_val, val_predictions))

        # Final evaluation on test set
        test_predictions = self.model.predict(X_test)
        print("Test Set Performance:")
        print(classification_report(y_test, test_predictions))

        return X_test, y_test

    def cross_validate(self):
        data = self.prepare_data()
        
        X_text = self.vectorizer.fit_transform(data['text'])
        X_features = data[['day_of_week', 'month', 'industry', 'exploit_count', 'remote_exploit_count']].values
        X = np.hstack((X_text.toarray(), X_features))
        y = data['target'].fillna(0).astype(int)

        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        print("Cross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())
        print("Standard deviation of CV score:", cv_scores.std())

        return cv_scores

    def time_based_validate(self):
        data = self.prepare_data()
        data = data.sort_values('published_at')
        
        X_text = self.vectorizer.fit_transform(data['text'])
        X_features = data[['day_of_week', 'month', 'industry', 'exploit_count', 'remote_exploit_count']].values
        X = np.hstack((X_text.toarray(), X_features))
        y = data['target'].fillna(0).astype(int)

        tscv = TimeSeriesSplit(n_splits=5)
        
        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            scores.append(score)
        
        print("Time-based cross-validation scores:", scores)
        print("Mean score:", np.mean(scores))
        print("Standard deviation of score:", np.std(scores))

        return scores

    def visualize_performance(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def analyze_feature_importance(self):
        feature_importance = self.model.feature_importances_
        feature_names = self.vectorizer.get_feature_names_out().tolist() + ['day_of_week', 'month', 'industry', 'exploit_count', 'remote_exploit_count']
        
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(12,8))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

        return importance_df

    def calibrate_probabilities(self, X_test, y_test):
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Plot calibration curve
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=f)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{self.model.__class__.__name__}")
        
        ax2.hist(y_pred_proba, range=(0, 1), bins=10, histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

    def predict(self, company_type):
        news_data = self.fetch_news()
        exploits_data = self.fetch_exploits()

        news_data['text'] = (news_data['title'].fillna('') + ' ' + news_data['summary'].fillna('')).apply(self.preprocess_text)
        news_data = self.extract_date_features(news_data)

        X_text = self.vectorizer.transform(news_data['text'])
        
        # Map user input to official category
        official_company_type = self.industry_mapping.get(company_type, company_type)

        # Handle unseen categories
        if official_company_type in self.label_encoder.classes_:
            industry_encoded = self.label_encoder.transform([official_company_type])[0]
        else:
            print(f"Warning: '{official_company_type}' is not in the trained categories. Using 'Undetermined' instead.")
            industry_encoded = self.label_encoder.transform(['Undetermined'])[0]
        
        # Check if 'date' column exists in exploits_data
        if 'date' in exploits_data.columns:
            exploit_count = news_data['published_at'].apply(
                lambda date: exploits_data[exploits_data['date'].dt.date == date.date()].shape[0]
            )
            remote_exploit_count = news_data['published_at'].apply(
                lambda date: exploits_data[(exploits_data['date'].dt.date == date.date()) & 
                                           (exploits_data['type'].str.contains('remote', case=False))].shape[0]
            )
        else:
            print("Warning: 'date' column not found in exploits_data. Setting exploit counts to 0.")
            exploit_count = pd.Series([0] * len(news_data))
            remote_exploit_count = pd.Series([0] * len(news_data))

        X_features = np.column_stack((
            news_data[['day_of_week', 'month']].values,
            np.full(news_data.shape[0], industry_encoded),
            exploit_count,
            remote_exploit_count
        ))

        X = np.hstack((X_text.toarray(), X_features))

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        news_data['prediction'] = predictions
        news_data['attack_probability'] = probabilities

        return news_data[['title', 'published_at', 'prediction', 'attack_probability']]

    def analyze_results(self, results, company_type):
        high_risk = results[results['attack_probability'] > 0.7]
        
        print(f"Analysis for {company_type} company:")
        print(f"Total predictions: {len(results)}")
        print(f"High-risk predictions: {len(high_risk)}")
        
        if not high_risk.empty:
            print("\nTop 5 high-risk predictions:")
            print(high_risk.sort_values('attack_probability', ascending=False).head())
        
        # Time series of attack probabilities
        plt.figure(figsize=(12,6))
        plt.plot(results['published_at'], results['attack_probability'])
        plt.title(f'Attack Probability Over Time for {company_type}')
        plt.xlabel('Date')
        plt.ylabel('Attack Probability')
        plt.show()

def main():
    predictor = CyberAttackPredictor()

    # Train and evaluate the model
    print("Training and evaluating the model...")
    X_test, y_test = predictor.train_and_evaluate()

    # Perform cross-validation
    print("\nPerforming cross-validation...")
    predictor.cross_validate()

    # Perform time-based validation
    print("\nPerforming time-based validation...")
    predictor.time_based_validate()

    # Visualize model performance
    print("\nVisualizing model performance...")
    predictor.visualize_performance(X_test, y_test)

    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    predictor.analyze_feature_importance()

    # Calibrate probabilities
    print("\nCalibrating probabilities...")
    predictor.calibrate_probabilities(X_test, y_test)

    # Make predictions for a specific company type
    print("\nAvailable company types:")
    for common_name, official_name in predictor.industry_mapping.items():
        print(f"- {common_name}")
    print("- Other (for any other category)")

    company_type = input("Enter your company type from the list above: ")
    results = predictor.predict(company_type)
    predictor.analyze_results(results, company_type)

if __name__ == "__main__":
    main()
