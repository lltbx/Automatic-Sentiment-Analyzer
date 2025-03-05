from string import punctuation, digits
import numpy as np
import random

#==============================================================================
#===  Algorithms  =================================================================
#==============================================================================
def load_stopwords(filename='stopwords.txt'):
    """
    Load stop words from a file and return them as a set for efficient lookup.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        stopwords = set(word.strip() for word in f)
    return stopwords

# Load stop words when the module is imported
stopwords = load_stopwords()


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    
    loss  = max (0, 1 - label*( np.dot(feature_vector, theta) + theta_0))
    return loss 



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """
    loss_full =  np.maximum(0, 1 - labels*( np.dot(feature_matrix, theta) + theta_0))  
    average_loss = np.mean(loss_full)
    return average_loss

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    if label*(np.dot(current_theta,feature_vector) +current_theta_0 )<=0:
            updated_theta = current_theta + label * feature_vector
            updated_theta_0 = current_theta_0 + label
            tuple = (updated_theta,updated_theta_0)
            return tuple
    return  (current_theta,current_theta_0)


def perceptron(feature_matrix, labels, T):
   
    n_features = feature_matrix.shape[1]
    current_theta = np.zeros(n_features)
    current_theta_0 = 0           
    for t in range(T):
          for i in get_order(feature_matrix.shape[0]):     
              feature_vector = feature_matrix[i] 
              label = labels[i]
              current_theta , current_theta_0 = perceptron_single_step_update(feature_vector, label, current_theta ,current_theta_0)
    return (current_theta,current_theta_0)

def average_perceptron(feature_matrix, labels, T):
   
    n_features = feature_matrix.shape[1]
    current_theta = np.zeros(n_features)
    current_theta_0 = 0
    updates = 0
    theta_sum_0 = 0
    theta_sum = np.zeros(n_features)  
    for t in range(T):
          for i in get_order(feature_matrix.shape[0]):  
              feature_vector = feature_matrix[i] 
              label = labels[i]
              current_theta , current_theta_0 = perceptron_single_step_update(feature_vector, label, current_theta ,current_theta_0)
              theta_sum += current_theta
              theta_sum_0 += current_theta_0
              updates +=1
    avg_theta = theta_sum/updates
    avg_theta_0 = theta_sum_0/updates
    return (avg_theta,avg_theta_0)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    prediction = np.dot(theta, feature_vector)
    
    if label * (prediction + theta_0) <= 1:
        # Update theta with L2 regularization and the gradient term
        # θ = (1 - ηλ) θ + η y x
        theta_update = (1 - eta * L) * theta + eta * label * feature_vector
        
        # Update theta_0 without regularization (no (1 - ηλ) scaling
        theta_0_update = theta_0 + eta * label
    else:
        # Only apply L2 regularization to theta 
        theta_update = (1 - eta * L) * theta
        theta_0_update = theta_0
    
    return (theta_update, theta_0_update)



def pegasos(feature_matrix, labels, T, L):
    
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    n_features = feature_matrix.shape[1]  # Number of features (columns)
    n_samples = feature_matrix.shape[0]   # Number of data points (rows)
    theta = np.zeros(n_features)
    theta_0 = 0.0
    update_counter = 0
    
    for t in range(T):
        indices = get_order(feature_matrix.shape[0])
        
        for i in indices:
            update_counter += 1
            eta = 1 / np.sqrt(update_counter)
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = pegasos_single_step_update(
            feature_vector, label, L, eta, theta, theta_0
            )
    
    return (theta, theta_0)

#==============================================================================
#===  Classification  ================================================================
#==============================================================================

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    predictions = np.dot(feature_matrix, theta) + theta_0
    classifications = np.where(np.isclose(predictions, 0), -1, np.where(predictions > 0, 1, -1))
    
    return classifications


def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, **kwargs):
    """
    Trains a linear classifier and computes accuracy. The classifier is
    trained on the train data. The classifier's accuracy on the train and
    validation data is then returned.
    """
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)
    train_acc = accuracy(train_predictions, train_labels)
    val_acc = accuracy(val_predictions, val_labels)
    return (train_acc, val_acc)


def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=True):  # Default to True for stop word removal
    """
    Creates a dictionary of words from texts, excluding stop words if specified.

    Args:
        `texts` - a list of natural language strings.
        `remove_stopword` - Boolean to indicate whether to remove stop words (default: True).
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`, excluding stop words if remove_stopword is True.
    """
    global stopwords  
    indices_by_word = {}  
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if remove_stopword and word in stopwords:
                continue
            if word not in indices_by_word:
                indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        # Binarize the feature matrix (convert counts to 1/0 presence)
        feature_matrix[feature_matrix > 0] = 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
