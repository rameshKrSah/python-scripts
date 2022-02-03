
import numpy as np
from sklearn import metrics as M
import pprint

def print_metrics(met_dict):
    """
    	Given a metrics dictionary, print the values for:
        - Loss
        - Accuracy
        - Precision
        - Recall
        - F1
    """
    print("Loss: {:.3f}".format(met_dict['Loss']))
    print("Accuracy: {:.3f} %".format(met_dict['Accuracy'] * 100))
    print("Precision: {:.3f}".format(met_dict['Precision']))
    print("Recall: {:.3f}".format(met_dict["Recall"]))
    print("F1 binary: {:.3f}".format(met_dict['F1_binary']))
    print("F1 macro: {:.3f}".format(met_dict['F1_macro']))

def compute_performance_metrics(y_true, y_pred):
  """ Compute precision, recall, and f1 score.
    y_true: true label
    y_pred: predicted label 
    
    Labels are not hot encoded
    
    Return a dictionary containing Accuracy, Precision, Recall, F1 Score (samples, macro, weighted) and Hamming Loss
  """
  # whether binary or multi-class classification
  if len(np.unique(y_true)) == 2:
    average_case = 'binary'
  else:
    average_case = 'macro'
  
  scores = {
        "Accuracy": M.accuracy_score(y_true, y_pred),
        "Precision": M.precision_score(y_true, y_pred, average=average_case),
        "Recall": M.recall_score(y_true, y_pred, average=average_case),
        "F1_samples": M.f1_score(y_true, y_pred, average="samples"),
        "F1_macro": M.f1_score(y_true, y_pred, average="macro"),
        "F1_micro": M.f1_score(y_true, y_pred, average="micro"),
        "F1_weighted": M.f1_score(y_true, y_pred, average="weighted"),
        "F1_binary": M.f1_score(y_true, y_pred, average="binary"),
        "Hamming Loss": M.hamming_loss(y_true, y_pred),
    }

  return scores


def compute_performance_metrics_binary(model, x, y, metric_names):
    """
        Given a model (TensorFlow) and (x, y). 
        
        Compute accuracy, loss, True Positive, False Negative, False Positive, True Negative, Recall, Precision, 
        f1 score, Average Precision Recall, ROC AUC, and classification report. Only for binary classification

        Arguments:
            model: tensorflow model
            x: feature vector
            y: label vector (one hot encoded)

        Returns: A dictionary containing, Accuracy, Loss, True Positive, False Positive, False Negative, 
                True Negative, Recall, Precision, f1 score, roc_auc_score
    """
    y_true = np.argmax(y, axis=1)
    if len(np.unique(y_true)) > 2:
      print("This only works for binary classification")
      return {}

    # get the metrics  
    metrics = model.evaluate(x, y)
    
    rt = dict()
    for name, val in zip(metric_names, metrics):
      rt[name] = val
  
    # the loss is always at first position and accuracy the second
    loss, acc = metrics[0], metrics[1] * 100 
    print("Accuracy {:.3f}, Loss {:.3f}".format(acc, loss))

    y_probs = model.predict(x)
    y_pred = np.argmax(y_probs, axis=1)

    tp, fp, tn, fn = (0, 0, 0, 0)

    try:
      # we can only do this in binary case
      tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except:
      print("Not a binary classification problem")
    
    print("True Positive ", tp)
    print("False Positive ", fp)
    print("True Negative ", tn)
    print("False Negative ", fn)

    recall = M.recall_score(y_true, y_pred)
    precision = M.precision_score(y_true, y_pred)

    print("Recall {:.3f}, with formula {:.3f}".format(recall, (tp / (tp + fn))))
    print("Precision {:.3f}, with formula {:.3f}".format(precision, (tp / (tp + fp))))

    f1_score_cal = M.f1_score(y_true, y_pred)
    print("F1 score {:.3f}, with formula {:.3f}".format(f1_score_cal,
           2 * ((precision * recall) / (precision + recall))))

    print("Average precision score {:.3f}".format(M.average_precision_score(y_true, y_pred)))

    roc_auc = M.roc_auc_score(y_true, y_pred)
    print("ROC AUC Score {:.3f}".format(roc_auc))
    
    clf_report = M.classification_report(y_true, y_pred, output_dict=True)
    pprint.pprint(clf_report)

    rt_dict = {'Accuracy': acc,
            'Loss': loss,
            'True Positive': tp, 
            'False Positive': fp, 
            'True Negative': tn, 
            'False Negative': fn,
            'Recall': recall,
            'Precision': precision,
            'F1 Score': f1_score_cal,
            'ROC AUC': roc_auc
            }

    return rt_dict
