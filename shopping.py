import csv
import sys


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    
    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    
    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    import pandas as pd 
    evidence=[]
    labels=[]
    data = pd.read_csv("shopping.csv") 
    
    data["Administrative"]= data["Administrative"].astype(int) 
    data["Administrative_Duration"]= data["Administrative_Duration"].astype(float) 
    data["Informational"]= data["Informational"].astype(float)
    data["Informational_Duration"]= data["Informational_Duration"].astype(float)
    data["ProductRelated"]= data["ProductRelated"].astype(int)
    data["ProductRelated_Duration"]= data["ProductRelated_Duration"].astype(float)
    data["BounceRates"]= data["BounceRates"].astype(float)
    data["ExitRates"]= data["ExitRates"].astype(float)
    data["PageValues"]= data["PageValues"].astype(float)
    data["SpecialDay"]= data["SpecialDay"].astype(float)
    look_up_month = {'Jan': '00', 'Feb': '01', 'Mar': '02', 'Apr': '03', 'May': '04',
            'June': '05', 'Jul': '06', 'Aug': '07', 'Sep': '08', 'Oct': '09', 'Nov': '10', 'Dec': '11'} 
    data["Month"]= data["Month"].apply(lambda x: look_up_month[x])
    data["Month"]= data["Month"].astype(int)
    data["OperatingSystems"]= data["OperatingSystems"].astype(int)
    data["Browser"]= data["Browser"].astype(int)
    data["TrafficType"]= data["TrafficType"].astype(int)
    look_up_VisitorType={'New_Visitor':'0', 'Returning_Visitor':'1', 'Other':'2'}
    data["VisitorType"]= data["VisitorType"].apply(lambda x: look_up_VisitorType[x])
    data["VisitorType"]= data["VisitorType"].astype(int)
    data["Weekend"]= data["Weekend"].astype(int)
    data["Revenue"]= data["Revenue"].astype(int)
    print(f"Fetching evidence and labels...")
    for index, rows in data.iterrows(): 
        evidence_list =[rows.Administrative, rows.Administrative_Duration, rows.Informational,rows.Informational_Duration,rows.ProductRelated,rows.ProductRelated_Duration,rows.BounceRates,rows.ExitRates,rows.PageValues,rows.SpecialDay,rows.Month,rows.OperatingSystems,rows.Browser,rows.Region,rows.TrafficType,rows.VisitorType,rows.Weekend]
        label=rows.Revenue
        evidence.append(evidence_list)
        labels.append(label)
    
    return evidence, labels
    


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels)
    return neigh
    


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = float(0)
    specificity = float(0)
    if labels.count(1)==0:
        sys.exit("No positve label in true labels")
    if labels.count(0)==0:
        sys.exit("No negative label in true labels")


    common_ones = [1 if i==j and j==1 else 0 for i, j in zip(labels,predictions)]
    common_ones_count=common_ones.count(1)
    labels_ones_count=labels.count(1)
    sensitivity=common_ones_count/labels_ones_count
    common_zeros=[1 if i==j and j==0 else 0 for i,j in zip(labels,predictions)]
    common_zeros_count=common_zeros.count(1)
    labels_zeros_count=labels.count(0)
    specificity=common_zeros_count/labels_zeros_count

    return sensitivity, specificity


    #raise NotImplementedError


if __name__ == "__main__":
    main()
