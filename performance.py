from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# %matplotlib inline

def run_performance(model, test_x, test_y):
    print 'Test Accuracy: ', model.score(test_x, test_y)

    print 'Confution matrix \n', confusion_matrix(test_y, model.predict(test_x))
    target_names = ['negative', 'positive']
    print(classification_report(test_y, model.predict(test_x), target_names=target_names))
    pred_probas = model.predict_proba(test_x)[:,1]

    fpr,tpr,_ = roc_curve(test_y, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.show()
