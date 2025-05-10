## Project Description  
This project compares multiple machine learning, deep learning, and large language model (LLM) approaches for text classification. It includes models such as **Bayesian Classifier**, **XGBoost**, **Random Forest**, **LSTM**, **BiLSTM**, **BERT**, and **LLaMA3-8B (Instruct + LoRA)**. The repository also provides data preprocessing tools and result visualization.

---

### How to Run the Code

#### Execution Commands  
```bash
python bayes.py          # Bayesian classifier
python rf_code.py        # Random Forest
python xgboost_code.py   # XGBoost

python lstm.py           # LSTM
python bilstm.py         # BiLSTM

jupyter notebook llama3.ipynb  # LLaMA3-8B (Instruct + LoRA)
```

---

### Experimental Results  

We evaluated the performance of various models. The accuracy comparison is shown below:

| Model               | Accuracy |
|---------------------|----------|
| Bayesian            | 69%      |
| Random Forest       | 71%      |
| XGBoost             | 74%      |
| LSTM                | 72%      |
| BiLSTM              | 76%      |
| BERT                | 81%      |
| **LLaMA3-8B**       | **86%**  |

**Key Observations**:  
- **LLaMA3-8B** achieves the highest accuracy (**86%**), outperforming traditional models like Bayes (69%) and deep learning models like BiLSTM (76%).  
- BERT and BiLSTM show strong performance among deep learning methods.  
- Tree-based models (XGBoost, Random Forest) perform better than simpler classifiers.  

---

### Notes  
- Ensure `combined_data.csv` is in the correct directory.  
- For memory-intensive models (e.g., LLaMA3-8B), use a GPU or reduce batch sizes.  
- Tokenizer and vectorizer files (`*.pkl`) are pre-trained; no retraining is required.  

--- 
