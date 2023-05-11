def prepare_data():
    import pandas as pd
    print("---- Inside prepare_data component ----")
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv")
    df = df.dropna()
    df.to_csv(f'data/final_df.csv', index=False)
    print("\n ---- data csv is saved to PV location /data/final_df.csv ----")

def train_test_split():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    print("---- Inside train_test_split component ----")
    final_data = pd.read_csv(f'data/final_df.csv')
    target_column = 'class'
    X = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify = y, random_state=47)
    
    np.save(f'data/X_train.npy', X_train)
    np.save(f'data/X_test.npy', X_test)
    np.save(f'data/y_train.npy', y_train)
    np.save(f'data/y_test.npy', y_test)
    
    print("\n---- X_train ----")
    print("\n")
    print(X_train)
    
    print("\n---- X_test ----")
    print("\n")
    print(X_test)
    
    print("\n---- y_train ----")
    print("\n")
    print(y_train)
    
    print("\n---- y_test ----")
    print("\n")
    print(y_test)

def training_basic_classifier():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    
    print("---- Inside training_basic_classifier component ----")
    
    X_train = np.load(f'data/X_train.npy',allow_pickle=True)
    y_train = np.load(f'data/y_train.npy',allow_pickle=True)
    
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train,y_train)
    import pickle
    with open(f'data/model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    print("\n logistic regression classifier is trained on iris data and saved to PV location /data/model.pkl ----")


def predict_on_test_data():
    import pandas as pd
    import numpy as np
    import pickle
    print("---- Inside predict_on_test_data component ----")
    with open(f'data/model.pkl','rb') as f:
        logistic_reg_model = pickle.load(f)
    X_test = np.load(f'data/X_test.npy',allow_pickle=True)
    y_pred = logistic_reg_model.predict(X_test)
    np.save(f'data/y_pred.npy', y_pred)
    
    print("\n---- Predicted classes ----")
    print("\n")
    print(y_pred)


def predict_prob_on_test_data():
    import pandas as pd
    import numpy as np
    import pickle
    print("---- Inside predict_prob_on_test_data component ----")
    with open(f'data/model.pkl','rb') as f:
        logistic_reg_model = pickle.load(f)
    X_test = np.load(f'data/X_test.npy',allow_pickle=True)
    y_pred_prob = logistic_reg_model.predict_proba(X_test)
    np.save(f'data/y_pred_prob.npy', y_pred_prob)
    
    print("\n---- Predicted Probabilities ----")
    print("\n")
    print(y_pred_prob)

def get_metrics():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    print("---- Inside get_metrics component ----")
    y_test = np.load(f'data/y_test.npy',allow_pickle=True)
    y_pred = np.load(f'data/y_pred.npy',allow_pickle=True)
    y_pred_prob = np.load(f'data/y_pred_prob.npy',allow_pickle=True)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    entropy = log_loss(y_test, y_pred_prob)
    
    y_test = np.load(f'data/y_test.npy',allow_pickle=True)
    y_pred = np.load(f'data/y_pred.npy',allow_pickle=True)
    print(metrics.classification_report(y_test, y_pred))
    
    print("\n Model Metrics:", {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)})
