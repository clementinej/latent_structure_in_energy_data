import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, precision_recall_curve, recall_score
#from wordcloud import WordCloud


BIZ_NAME = 'last_name'
TRAINING = 'training_set'
SEGMENT = 'segment_name'
SUBTYPE1 = 'subtype1'
SUBTYPE2 = 'subtype2'


def load_business_types(path='data/smb_customers.tsv'):
    biz = pd.read_csv(path, sep='\t').dropna(how='all')
    return biz.loc[(biz[BIZ_NAME].notnull()) & (biz[SEGMENT] != 'Excluded')]


def create_features(biz):
    """ Takes training data only """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(biz[BIZ_NAME])
    return X, vectorizer


def plot_wordcloud(results):
    wordcloud = WordCloud(font_path='/System/Library/Fonts/Amble-Regular.ttf', background_color='white')
    word_threshold_index = min(100, len(results['top_features'].values())-1)
    word_threshold = sorted(results['top_features'].values(), reverse=True)[word_threshold_index]
    top_words = ((word, value) for word, value in results['top_features'].iteritems() if value > word_threshold)
    wordcloud.fit_words(list(top_words))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def run_classifier(biz_features, biz_labels,
                   label,
                   test_features, test_labels,
                   words, word_cloud_image='data/Pluggy.png',
                   estimator_type='RandomForest',
                   parameters=None):

    y = biz_labels.map(lambda x: x == label)
    y_test = test_labels.map(lambda x: x == label)

    if estimator_type == 'RandomForest':
        default_parameters = {'n_estimators': [100]}
        estimator = RandomForestClassifier(n_jobs=-1)
    elif estimator_type == 'LogisticRegression':
        default_parameters = {'tol': [1e-4], 'C': [1.0], 'penalty': ['l2']}
        estimator = LogisticRegression()
    else:
        assert False, "Unrecognized estimator %s" % estimator_type

    if parameters is not None:
        default_parameters.update(parameters)
    classifier = GridSearchCV(estimator, default_parameters, cv=10)
    classifier.fit(X=biz_features.toarray(), y=y)
    classes = classifier.best_estimator_.classes_
    best_score = classifier.best_score_
    if estimator_type == 'RandomForest':
        importances = classifier.best_estimator_.feature_importances_
    elif estimator_type == 'LogisticRegression':
        importances = classifier.best_estimator_.coef_[0]

    probabilities = classifier.predict_proba(test_features.toarray())[:, np.where(classes == True)].ravel()
    predicted = classifier.predict(test_features.toarray())
    #print "Predicted shape: {}".format(predicted.shape)
    precision_curve, recall_curve, threshold = precision_recall_curve(y_test, probabilities)
    precision = precision_score(y_test, predicted)
    recall = recall_score(y_test, predicted)

    word_importances = {w: importances[idx] for w, idx in words.iteritems() if importances[idx] > 0.}
    # word cloud

    return {
        'label': label,
        'label %': y.mean(),
        'classifier': classifier,
        'cv_score': best_score,
        'top_features': word_importances,
        'precision': precision,
        'recall': recall,
        'precision_recall_curve-precision': precision_curve,
        'precision_recall_curve-recall': recall_curve,
        'precision_recall_curve-threshold': threshold
    }


def print_business_type_results(results, business_type):
    precision_threshold = 0.95
    precision_curve = results['precision_recall_curve-precision']
    recall_curve = results['precision_recall_curve-recall']

    fig, ax = plt.subplots(1)
    plt.plot(recall_curve, precision_curve)
    precision_index = find_precision_threshold_index(precision_curve, recall_curve, precision_threshold)
    plt.axvline(recall_curve[precision_index], color='red')
    plt.axhline(precision_threshold, color='red')
    plt.text(recall_curve[precision_index], plt.axis()[2] + 0.05, 'Optimal prediction threshold', color='red',
             verticalalignment='baseline', size='larger', weight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(business_type)
    plt.show()

    print "Chosen model: %s" % results['classifier'].best_estimator_
    # TODO: the baseline is based on the training labels and should be based on the test labels
    print ("Overall predictive power: {0:.1f}% (baseline predictive power is {1:0.1f}%)"
           .format(results['cv_score'] * 100, max(results['label %']-1, 1-results['label %']) * 100))

    if precision_index > 0:
        recall_meeting_threshold = recall_curve[precision_index]
        precision_meeting_threshold = precision_curve[precision_index]
        print ("We can identify {0:0.1f}% of {1} with at least 95% confidence, which means we can send reports "
               "to an extra {2:0.0f} customers that were previously unaddressable for Xcel Minnesota and Xcel Colorado"
               .format(recall_meeting_threshold*100, business_type, # precision_meeting_threshold*100,
                       results['label %']*400000*recall_meeting_threshold))


def find_precision_threshold_index(precision_curve, recall_curve, precision_threshold):
    return pd.Series(recall_curve).where(precision_curve>=precision_threshold).argmax()
