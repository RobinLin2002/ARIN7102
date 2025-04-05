# Import Dependencies
import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
# Feature selection
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.svm import SVC


class DiseasePrediction:
    # Initialize and Load the Config File：读取配置文件加载参数
    def __init__(self, model_name=None):
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Verbose
        self.verbose = self.config['verbose']
        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Feature Selection
        self.train_features, selected = self._feature_selection(self.train_features, self.train_labels)
        self.test_features = self.test_features[selected]  # apply same features to test set
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_features, show_fig=False)
    
        # Model Definition
        self.model_name = model_name
        # Model Save Path
        self.model_save_path = self.config['model_save_path']

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    # Function to Load Test Dataset
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data sanity
        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    def _feature_selection(self, X, y, var_thresh=0.0, k_best=100, corr_thresh=0.95):
        # Step 1: Variance Threshold
        vt = VarianceThreshold(threshold=var_thresh)
        vt.fit(X)
        selected_var = X.columns[vt.get_support()]
        print("Step 1 - Features after Variance Thresholding:\n", list(selected_var), "\n")
        
        # Step 2: Chi-square test
        chi = SelectKBest(score_func=chi2, k=min(k_best, len(selected_var)))
        chi.fit(X[selected_var], y)
        selected_chi = selected_var[chi.get_support()]
        print("Step 2 - Features after Chi-square Test:\n", list(selected_chi), "\n")
        
        # Step 3: Remove high-correlation features
        corr = X[selected_chi].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
        final_features = selected_chi.drop(to_drop)
        print("Step 3 - Final Features after Correlation Filtering:\n", list(final_features), "\n")
        if self.verbose:
            print(f"Selected {len(final_features)} features after feature selection.")
        return X[final_features], final_features

    # Features Correlation
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    # Dataset Train Validation Split：从训练集里面划分验证集
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # Model Selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'svm':
            self.clf = SVC(probability=True)
        return self.clf

    # ML Model
    def train_model(self):
        # Get the Data：划分训练集和验证集
        X_train, y_train, X_val, y_val = self._train_val_split()
        # 选择model
        classifier = self.select_model()
        # Training the Model：模型训练
        classifier = classifier.fit(X_train, y_train)
        # Trained Model Evaluation on Validation Dataset：在验证集上评估
        train_acc = classifier.score(X_train, y_train)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        val_acc = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix：左边是实际label，上边是预测的label，所以对角线上是预测对的个数，非对角线就都是预测错的个数
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report：输出precision，recall，F1等
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score：三折交叉验证，判断模型有没有overfit或者对划分敏感
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', train_acc)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', val_acc)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    # Function to Make Predictions on Test Data：在测试集上做预测
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model：加载训练好的模型
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")

        if test_data is not None: #用户输入新的test data，返回预测结果
            # result = clf.predict(test_data)
            # return result
            # 修改成输出所有大于10%概率的疾病
            probs = clf.predict_proba(test_data)
            class_names = clf.classes_
            prob = probs[0]
            threshold = 0.2
            filtered = [(cls, round(p, 4)) for cls, p in zip(class_names, prob) if p>=threshold]
            filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
            return filtered
        else:
            result = clf.predict(self.test_features) #没有的话直接用test_feature里面的feature，返回准确率和report
        accuracy = accuracy_score(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        return accuracy, clf_report


if __name__ == "__main__":
    # Model Currently Training
    current_model_name = 'svm'
    # Instantiate the Class
    dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    dp.train_model()
    # Get Model Performance on Test Data
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    print("Model Test Accuracy: ", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)