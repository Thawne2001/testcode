#import kfp libraries
import kfp
import kfp.components as comp
import requests
import kfp.dsl as dsl

#define functions
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

#Create components 
create_step_prepare_data = kfp.components.create_component_from_func(
    func=prepare_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0']
)

create_step_train_test_split = kfp.components.create_component_from_func(
    func=train_test_split,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_training_basic_classifier = kfp.components.create_component_from_func(
    func=training_basic_classifier,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_predict_on_test_data = kfp.components.create_component_from_func(
    func=predict_on_test_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_predict_prob_on_test_data = kfp.components.create_component_from_func(
    func=predict_prob_on_test_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_get_metrics = kfp.components.create_component_from_func(
    func=get_metrics,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)


# Define the pipeline
@dsl.pipeline(
   name='IRIS classifier Kubeflow Demo Pipeline',
   description='A sample pipeline that performs IRIS classifier task'
)
# Define parameters to be fed into pipeline
def iris_classifier_pipeline(data_path: str):
    vop = dsl.VolumeOp(
    name="t-vol",
    resource_name="t-vol", 
    size="1Gi", 
    modes=dsl.VOLUME_MODE_RWO)
    
    prepare_data_task = create_step_prepare_data().add_pvolumes({data_path: vop.volume})
    train_test_split = create_step_train_test_split().add_pvolumes({data_path: vop.volume}).after(prepare_data_task)
    classifier_training = create_step_training_basic_classifier().add_pvolumes({data_path: vop.volume}).after(train_test_split)
    log_predicted_class = create_step_predict_on_test_data().add_pvolumes({data_path: vop.volume}).after(classifier_training)
    log_predicted_probabilities = create_step_predict_prob_on_test_data().add_pvolumes({data_path: vop.volume}).after(log_predicted_class)
    log_metrics_task = create_step_get_metrics().add_pvolumes({data_path: vop.volume}).after(log_predicted_probabilities)

    
    prepare_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_test_split.execution_options.caching_strategy.max_cache_staleness = "P0D"
    classifier_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_class.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_probabilities.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_metrics_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

#create yaml file
kfp.compiler.Compiler().compile(
    pipeline_func=iris_classifier_pipeline,
    package_path='IRIS_Classifier_pipeline1.yaml')
