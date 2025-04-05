training data: 
总共4920条数据, # of features (症状): 134： output：可能的疾病 （总共42种）
为了确保训练数据质量，每个疾病种类对应的总共数据量都是均等的

test data: 
数量42 （对应每一种疾病）

feature selection过程: 1. Variance Thresholding 2.Chi-square Test 3.Correlation Filtering

结果
SVM: 
Training Accuracy:  0.910497572815534
Validation Accuracy:  0.9193349753694581
Model Test Accuracy:  0.9047619047619048

MNB:
Training Accuracy:  0.8837985436893204
Validation Accuracy:  0.8996305418719212
Model Test Accuracy:  0.8809523809523809

RandomForest:
Training Accuracy:  0.910497572815534
Validation Accuracy:  0.9193349753694581
Model Test Accuracy:  0.9047619047619048

Decision Tree
Training Accuracy:  0.910497572815534
Model Test Accuracy:  0.9047619047619048
Validation Accuracy:  0.9193349753694581


 Model Performance on Selected Features:

        Model  Accuracy  Precision   Recall  F1 Score  ROC AUC
MultinomialNB  0.880952   0.870732 0.890244  0.869919 0.991307
 DecisionTree  0.904762   0.896341 0.914634  0.895935 0.989426
 RandomForest  0.904762   0.896341 0.914634  0.895935 0.989723
          SVM  0.904762   0.896341 0.914634  0.895935 0.996431