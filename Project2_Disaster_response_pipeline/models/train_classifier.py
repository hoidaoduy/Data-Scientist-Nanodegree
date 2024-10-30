import sys
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    """
    Purpose: Loads data from SQLite database.
    
    Input:
    database_filepath: Filepath to the SQLite database.
    
    Output:
    X: Features for training (messages).
    Y: Target labels for each category.
    category_names: List of category names (target labels).
    """
    # Load data from database 
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table("disaster_messages", con=engine)
    
    # Extract feature and target variables
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Purpose: Tokenizes and lemmatizes text.
    
    Input:
    text: Text to be tokenized
    
    Output:
    clean_tokens: Returns cleaned tokens 
    """
    # Convert to lowercase, replace non-letter/number characters with spaces, and remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize text using word_tokenize of nltk
    tokens = word_tokenize(text)
    
    # Initiate lemmatizer from nltk
    lemmatizer = WordNetLemmatizer()
    
    # Iterate through each token
    clean_tokens=[]
    for tok in tokens:
        # Lemmatize, normalise case, and remove white space in leading and trailing
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Purpose: Builds a machine learning pipeline and tunes the model's hyperparameters using GridSearchCV.

    Input: 
    None

    Output:
    gscv: GridSearchCV object, a classifier pipeline wrapped in GridSearchCV, configured to find the best combination of hyperparameters.

    Description:
    The function constructs a machine learning pipeline that performs text vectorization, TF-IDF transformation, 
    and classification using a multi-output RandomForestClassifier. It then applies GridSearchCV to tune parameters, 
    optimizing the model based on specified options for TF-IDF usage, number of estimators, and minimum sample splits 
    in the RandomForestClassifier.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 4]
    }
    
    gscv = GridSearchCV(pipeline, param_grid=parameters)
    
    return gscv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Purpose: Evaluates the performance of model and returns classification report. 
    
    Input:
    * model: classifier
    * X_test: test dataset
    * Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, ": ",classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ 
    Purpose: Exports the final model as a pickle file.

    Input:
    model: Trained model object, typically the result of a GridSearchCV or a trained estimator.
    model_filepath: String path to the location where the model will be saved as a .pkl file.

    Output:
    Saves the model to the specified filepath in pickle format for future use.
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()