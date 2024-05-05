from __future__ import annotations

from math import ceil
import re
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, \
                            multilabel_confusion_matrix, auc

# Dependencies to import only for type checking.
if TYPE_CHECKING:
    from keybert import KeyBERT
    import sklearn
    from sklearn.pipeline import Pipeline
    import spacy

def is_string_series(s: pd.Series) -> bool:
    """
    Checks whether s series contains only strings.

    Parameters
    ----------
    s : pd.Series
        Series to be checked.
    """
    if isinstance(s.dtype, pd.StringDtype):
        # String series.
        return True

    elif s.dtype == 'object':
        # Object series --> must check each value.
        return all((v is None) or isinstance(v, str) for v in s)

    else:
        return False


def categories_as_lists(df: pd.DataFrame) -> None:
    """
    Ensures that the category column contains lists of strings.

    Parameters
    ----------
    df : pd.DataFrame
         Dataframe to be checked.
    """
    if(is_string_series(df["category"]) == True):
        df["category"] =  df["category"].map(eval)
      
    return


def categories_as_strings(df: pd.DataFrame) -> None:
    """
    Ensures that the category column contains strings.

    Parameters
    ----------
    df : pd.DataFrame
         Dataframe to be checked.
    """
    if(is_string_series(df["category"]) == False):
        df["category"] =  df["category"].map(str)
      
    return


def plot_df_counts(df: pd.DataFrame, col: str) -> dict:
    """
    Computes the occurrences of the lists in the col column
    of the df dataframe and plots histograms. Everything is
    repeated for the exploded dataframe.

    Parameters
    ---------
    df : pd.DataFrame
         Dataframe that contains the column col.
    col : string
          Name of the column whose elements must be counted.

    Returns
    -------
    dict_counts : dictionary 
                  Its keys are the names contained in col's lists 
                  and its values are their occurrences.
    """

    # Before grouping check if we have strings.
    categories_as_strings(df)

    # Grouping by the 'col' column and counting the occurrences.
    df_counts = df[[col]].groupby([col])[col].count()
    df_counts = df_counts.reset_index(name='counts')

    # Sorting in descending order.
    df_counts = df_counts.sort_values(['counts'], ascending=False)
    df_counts = df_counts.reset_index(drop=True)

    # Creating a dictionary.
    names = df_counts[col].tolist()
    counts = df_counts['counts'].tolist()
    dict_counts = dict([(v, c) for v, c in zip(names, counts)])

    # Plot the histogram.
    df_counts.plot.bar(x=col, y='counts', 
                       color='r', figsize=(20,5))
  
    return dict_counts


def run_model(pipeline: sklearn.pipeline.Pipeline, 
              X_train: pd.Series, X_test: pd.Series, 
              y_train: np.ndarray, y_test: np.ndarray, 
              multilabel: bool) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Fit the data and predict a classification.

    Parameters
    ---------
    pipeline : sklearn.pipeline.Pipeline
               Pipeline that perform the vectorization and the classification on
               the given data.
    X_train : pd.Series
              Section of the data that are used as train features.
    X_test : pd.Series
             Section of the data that are used as test features.
    y_train : numpy.ndarray
              Section of the data that are used as train labels.
    y_test : numpy.ndarray
             Section of the data that are used as test labels.
    multilabel : bool
                 True if the prediction is multilabel,
                 False otherwise.

    Returns
    -------
    y_pred : numpy.ndarray
             Predictions of the test data.
    mat : numpy.ndarray
          Confusion matrices.
    """
    # Fit of the train data using the pipeline.
    pipeline.fit(X_train, y_train)
    # Prediction on the test data.
    y_pred = pipeline.predict(X_test)

    if multilabel:
        # Compute the confusion matrices.
        mat = multilabel_confusion_matrix(y_test, y_pred)
        return y_pred, mat
        
    else:
        return y_pred
    

def text_cleaner(text: str, nlp: spacy.lang.en.English) -> str:
    """
    After joining interrupted words and tokenizing the text, 
    lemmatize, remove bad words, special characters, punctuation
    and Spacy stopwords.

    Parameters
    ---------
    text : string
           Text to be cleaned.
    nlp : spacy.lang.en.English
          Spacy model.

    Returns
    -------
    clean_tokens : string
                   Cleaned text.
    """
    # Join interrupted words.
    text = text.replace("- ", "")  

    # Tokenize.
    tokens = nlp(text) # list of words

    # Lemmatize and transform to lowercase.
    #"-PRON-" is used as the lemma for all pronouns such as I, me, their ...
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" 
              else word.lower_ for word in tokens] 

    # Remove the tokens that contain any string in badlist.
    badlist = [".com", "http"] 
    clean_tokens = [word for word in tokens 
                    if not any(bad in word for bad in badlist)]
  
    # Remove stopwords.
    from spacy.lang.en.stop_words import STOP_WORDS
    clean_tokens = [word for word in clean_tokens 
                    if not word in STOP_WORDS] 

    # Keep only characters that are alphabet letters or spaces.
    clean_tokens = " ".join(c for c in clean_tokens 
                            if c.isalpha() or c.isspace()) 
      
    return clean_tokens


def plot_confusion_matrices(mat: np.ndarray, classes: np.ndarray) -> None:
    """
    Plot the confusion matrices normalizing on columns.

    Parameters
    ---------
    mat : np.ndarray
          Confusion matrices given by the classification.
    classes : np.ndarray
              All the possible categories.      
    """
    num_mat = len(mat) # number of confusion matrices we want to plot
    
    # Find the number of cells in the grid that will contain the num_mat subplots.
    num_rows = ceil(np.sqrt(num_mat))
    num_cols = ceil(num_mat/num_rows)
    num_cells = num_rows*num_cols
    rest = num_cells - num_mat

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,20))
    axes = axes.ravel() # get flattened axes

    # Iterate over the cells.
    for i in range(num_cells):

        if i < num_mat:
            # Plot the matrix.
            disp = ConfusionMatrixDisplay(normalize(mat[i], axis=0, norm='l1'))
            disp.plot(ax=axes[i])
            disp.ax_.set_title(f'{classes[i]}')

            # Only show x and y labels for the plots in the border.
            first_i_of_last_row = num_rows*num_cols - num_cols
            if i < first_i_of_last_row - rest:
                disp.ax_.set_xlabel('') # do not set the x label

            is_i_in_first_col = i%num_cols == 0
            if is_i_in_first_col == False:
                disp.ax_.set_ylabel('') # do not set the y label

            disp.im_.colorbar.remove() # remove it to put it after

        else: # delete axes in excess
            fig.delaxes(axes[i])


    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)
  
    return


def ROC(classes: np.ndarray, y_test: np.ndarray, y_score: np.ndarray) -> None:
    """
    Plot the ROC curves and compute their areas.

    Parameters
    ---------
    classes : np.ndarray
              All the possible categories.  
    y_test : np.ndarray
             Section of the data that are used as test labels.
    y_score : np.ndarray
              Decision function of X_test.      
    """

    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Sort the dictionary based on the area value.
    roc_auc_ord = dict(sorted(roc_auc.items(), key=lambda item: item[1]))

    # Take the sorted indices.
    indices = list(roc_auc_ord.keys())

    # Plot ROC curve.
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle('color', [cm(1.*i/n_classes) for i in range(n_classes)])

    for i in indices:
        ax.plot(fpr[i], tpr[i], label='{0} (area = {1:0.2f})'
                                      ''.format(classes[i], roc_auc[i]))


    # Compute micro-average ROC curve and ROC area.
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ax.plot(fpr["micro"], tpr["micro"], color='k',
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]))

    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curves')
    plt.legend(loc="lower right", fontsize='8', framealpha=0.5, ncol=2)
  
    return


def extract_kws(text: str, kw_model: keybert._model.KeyBERT, seed: List[str], top_n: int) -> List[str]:
    """
    Extract a list of top_n keywords for the input text using 
    some seed-keywords given by seed.

    Parameters
    ---------
    text : string
           Text from which to extract keywords.
    kw_model : keybert._model.KeyBERT
               KeyBERT model.
    seed : list of strings
           Seed keywords that might guide the extraction of keywords.
    top_n : int
            Number of keywords to extract.

    Returns
    -------
    keywords: list of strings
              List of the top_n extracted keywords.
    """
  
    max_n_grams = 1
    if seed == ['']: # if there are no words to use as seeds
        seed = None # switch off seed_keywords parameter below
    data = kw_model.extract_keywords(docs=text,
                                     keyphrase_ngram_range=(1, max_n_grams),
                                     seed_keywords = seed,
                                     stop_words='english',
                                     use_mmr=True,
                                     top_n=top_n)
    keywords = list(list(zip(*data))[0])
  
    return keywords
