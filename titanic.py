# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import math
import re
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import collections
import operator as opr
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt;


def HandleAge(val) :
    return round( val/5, 0 );


def NamesGroupBy( val ):
    l = val.find(',');
    if( l == -1 ):
        return val;
    return val[:l];
    
def HandleName(val) :
    if( val.find( 'Mr.' ) != -1 or  val.find( 'Sir.' ) != -1 ):
        return 1;
    if(val.find( 'Master.' ) != -1) :
        return 3;
    if(val.find( 'Miss.' ) != -1 or val.find( 'Mlle.' ) != -1 ) :
        return 4;
    if( val.find( 'Mrs.' ) != -1 or val.find( 'Lady.' ) != -1 ):
        return 5;
    return 2;
    
dic = {}
currentCount = int(0);
def HandleTicket( val ):
    global dic;
    global currentCount;
    onlyDig = ""
    val = re.sub( "\d+ ", "", val[::-1]).strip();
    onlyDig = re.sub( "\d", "", val).strip();
    val = ''.join(e for e in val if e.isalnum())
    dic['d'] = 0;
    if( val.isspace() or onlyDig.isspace() or not onlyDig or not val) :
        return 0;
    op = dic.get(val, -1)
    if( op == -1 ) :
        currentCount=currentCount+1;
        dic[val] = currentCount;
        return currentCount;
    return op;

def HandleCabin(val):
    if( val is float ):
        return 0;
    valstr = str(val)
    if(valstr == 'nan'):
        return 0
    op = re.sub( "\d*", "", valstr ).strip()[0];
    return op

dicCabin = {}
def FindBestCabinProbs( cabinProbs ):
    global dicCabin;
    sortedCabinProbs = sorted(cabinProbs.items(), key=opr.itemgetter(1));
    counter= 0;
    for c in sortedCabinProbs:
        counter = counter + 1;
        dicCabin[c[0]] = counter;

def HandleCabin2(val):
    global dicCabin;
    return dicCabin[val];
    
def HandleFare(val):
    return round(val/30, 0);
    
def HandleEmbarked(val):
    if val == 'S':
        return 1;
    if val == 'Q':
        return 3;
    if val == 'C':
        return 2;
    return 0;

def CalculatePredictionEffeciancy(pred, out) :
    diff = pred == out;
    pmap = collections.Counter( diff )
    return pmap[True]/diff.size;

def CalculatePercentageOfSurvived( column, data, function  ):
    survivedPpl = data.loc[ data['Survived']==True ];
    if( function != None ) :
        survivedCol = survivedPpl[column].apply( function )
        allCol = data[column].apply(function)
    else:
        survivedCol = survivedPpl[column]
        allCol = data[column]
    c1 = collections.Counter(survivedCol)
    c2 = collections.Counter(allCol)
    ret = {}
    for c in c2 :
        val = c1.get(c, -1);
        if(val == -1):
            ret[c] = 0;
            continue;
        ret[c] = val/c2[c];
    return ret
    ''' TO BE DONE
def CalculateWeightedAverage( column, basic_data ):
    waDic = {}
    c2 = collections.Counter(basic_data[column]);
    total = len(data)
    for c in c2:
        waDic[c] =  c2[c]/total
    return waDic;

def MultipyWA( table, waTable, column ):
    waD = CalculateWeightedAverage( column,  waTable);
    for wa in waD:
'''        
    
def ProcessInputForMLP( inpt ):  
    inpMLP = inpt.copy();
    #weighted average
    '''
    inpMLP['Age'] = inpMLP['Age']/max(inpt['Age']);
    inpMLP['Ticket'] = inpMLP['Ticket']/(inpt['Ticket'].mean());
    inpMLP['Fare'] = inpMLP['Fare']/max(inpt['Fare']);
    inpMLP['Cabin'] = inpMLP['Cabin']/(inpt['Cabin'].mean());
    inpMLP['Embarked'] = inpMLP['Embarked']/(inpt['Embarked'].mean());
    '''
    scaler = StandardScaler()
    scaler.fit(inpMLP)
    inpMLP = scaler.transform(inpMLP)
    return inpMLP;

survivedGroupBy = None;
minAgeGroupBy = None;
def CalculateProbabilities( data, test ):
    global dic;
    global survivedGroupBy;
    global minAgeGroupBy;
    inp = data.copy()
    cabinProbs = CalculatePercentageOfSurvived( 'Cabin', inp, HandleCabin );
    FindBestCabinProbs(cabinProbs);
    
    """Retreat tickets according to probabilities"""
    tickProbs = CalculatePercentageOfSurvived( 'Ticket', inp, HandleTicket );
    sortedTickProbs = sorted(tickProbs.items(), key=opr.itemgetter(1));
    dic2 = {}
    counter = 0;
    for key in dic:
        dic2[dic[key]] = key;
        
    for val in np.array(sortedTickProbs)[:,0]:
        dic[ dic2[int(val)] ] = counter;
        counter = counter+1
    inp['Ticket'] = data['Ticket'].apply(HandleTicket)
    
    inp2 = data.copy()
    inp2['Age'] = inp2['Age'].fillna( data['Age'].median() )
    inp2['Name'] = inp2['Name'].apply(NamesGroupBy)
    inp2['Survived'] = inp2['Survived'].apply( float )
    survivedGroupBy = inp2.groupby( 'Name' )['Survived'].mean() * inp2.groupby( 'Name' )['Survived'].count();
    survivedGroupBy.columns = ['Name', 'S']
    
    inp3= data[ ['Name','Age'] ].copy();
    inp3 = [inp3, test[['Name','Age']]]
    inp3 = pd.concat(inp3);
    inp3['Name'] = inp3['Name'].apply(NamesGroupBy)
    inp3['Age'] = inp3['Age'].fillna( data['Age'].median() )
    minAgeGroupBy = inp3.groupby( 'Name' )['Age'].min()


def ProcessData( data ):
    
    global survivedGroupBy;
    global minAgeGroupBy;
    inp = data.copy();
    
    del inp['PassengerId']
    inp['Sex'] = inp['Sex'] == 'female';
    inp['Name'] = inp['Name'].apply(HandleName);
    inp['Age'] = inp['Age'].fillna( data['Age'].median() )
    inp['Age'] = inp['Age'].apply(HandleAge);
    inp['Ticket'] = inp['Ticket'].apply(HandleTicket)
    
    inp['Cabin'] = inp['Cabin'].apply(HandleCabin)
    inp['Cabin'] = inp['Cabin'].apply(HandleCabin2)
    
    inp['Fare'] = inp['Fare'].fillna( data['Fare'].median() )
    inp['Fare'] = inp['Fare'].apply(HandleFare)
    inp['Embarked'] = inp['Embarked'].fillna( 'S' )
    inp['Embarked'] = inp['Embarked'].apply(HandleEmbarked)
    
    '''
    inp2 = data.copy()
    inp2['Name'] = inp2['Name'].apply(NamesGroupBy)
    survivedRelativesScore = survivedGroupBy.get( inp2['Name'] );
    survivedRelativesScore = survivedRelativesScore.reset_index();
    survivedRelativesScore.columns = [ 'Name', 'SurvivedRelatives'];
    inp['SurvivedRelatives']=survivedRelativesScore['SurvivedRelatives'];
    inp['SurvivedRelatives'] = inp['SurvivedRelatives'].fillna(0.5);
    
    minFamilyAge = minAgeGroupBy.get( inp2['Name'] )
    
    minFamilyAge = minFamilyAge.reset_index();
    minFamilyAge.columns = [ 'Name', 'MinFamilyAge'];
    inp[ 'MinFamilyAge' ] = minFamilyAge['MinFamilyAge']
    '''    
    
    del inp['Ticket']
    del inp['Embarked']
    del inp['Name']    
    
    return inp;

def PlotDictionary(D):
    plt.bar(range(len(D)), D.values(), align='center')
    plt.xticks(range(len(D)), D.keys())
    
    plt.show()


data = pd.read_csv('/home/prat/Kaggle/titanic/train.csv');
testInp = pd.read_csv('/home/prat/Kaggle/titanic/test.csv');

CalculateProbabilities(data, testInp);
inp = ProcessData(data);
output = inp['Survived'].copy()
del inp['Survived']
passangerArray = testInp['PassengerId']

'''
minFamilyAge = minFamilyAge.reset_index()
minFamilyAge.columns = ['Name', 'Age']
inp['MinFamilyAge' ] = minFamilyAge['Age'];
inp['MinFamilyAge' ] = inp['MinFamilyAge' ].apply(float)
CalculatePercentageOfSurvived('MinFamilyAge' , inp, None )
'''


""" MLP """
inpMLP = pd.DataFrame( ProcessInputForMLP(inp) );
'''
clf = MLPClassifier(solver='lbfgs', activation = 'logistic',
                    alpha=1e-6,
                    hidden_layer_sizes=(7, 7), 
                    tol=0.0000000000000000000000000000001, 
                    random_state=1, 
                    max_iter= 1000000000000, 
                    early_stopping=False, 
                    learning_rate_init=0.0001, validation_fraction=0.05);

'''
clf = MLPClassifier(solver='lbfgs', activation = 'logistic',
                    alpha=1e-2,
                    hidden_layer_sizes=(100, 100), 
                    tol=1e-32, 
                    random_state=1, 
                    max_iter= 100000000000, 
                    early_stopping=False,
                    validation_fraction=0, verbose = True);
                    
clf.fit(inpMLP.as_matrix(), output.as_matrix());
pred = clf.predict(inpMLP.as_matrix());
CalculatePredictionEffeciancy(pred, output)
failed = data.loc[ pred!= output ]
failed2 = inp.loc[ pred!= output ]
#train more on failed data
specialize = inpMLP.loc[ pred!=output ];
outSpecialized = output.loc[ pred!=output ];
clf.fit(specialize.as_matrix(), outSpecialized.as_matrix());
pred = clf.predict(inpMLP.as_matrix());
CalculatePredictionEffeciancy(pred, output)
""" END """

""" SVM """
svmArr = inpMLP
#svmLearn = svm.NuSVC()
#svmLearn = svm.SVC(kernel = 'rbf', probability = True, tol = 1e-7,verbose=True, decision_function_shape = 'ovo')
svmLearn = svm.SVC();
svmLearn.fit(svmArr, output.as_matrix()) 
pred = svmLearn.predict(svmArr);
CalculatePredictionEffeciancy(pred, output)
""" SVM END """

#Random forests
rfcInp = inp.as_matrix();
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20)
rfc = rfc.fit(rfcInp, output.as_matrix());
pred = rfc.predict( rfcInp )
CalculatePredictionEffeciancy(pred, output)
failed = data.loc[ pred!= output ]
#random forest end


#ADA Boost Classifier
from sklearn.ensemble import AdaBoostClassifier
adaInp = inp.as_matrix()
ada = AdaBoostClassifier(svm.SVC(kernel='linear',tol = 1e-10),n_estimators=5,learning_rate=0.0000000001, algorithm='SAMME')
ada.fit( adaInp, output.as_matrix() )
ada.score(adaInp, output)
#

test = ProcessData( testInp )

scaler = StandardScaler()
scaler.fit(inpMLP)
testMLP = scaler.transform(test)

#random forest
rfcTest = test.as_matrix()
prediction = rfc.predict( rfcTest )

prediction = clf.predict(testMLP);
#prediction = svmLearn.predict(test);
res = np.vstack( (passangerArray, prediction > 0.5) )
respd = pd.DataFrame(res.T)
respd.columns = ['PassengerId', 'Survived']
respd.to_csv( '/home/prat/Kaggle/titanic/pred3.csv', index = False );

"""Plots for data
"""
survivedPpl = data.loc[ data['Survived']==True ]
survivedAge = survivedPpl['Age'].fillna( data['Age'].mean() )
survivedAge = survivedAge.apply(HandleAge)

allAge = data['Age'].fillna( data['Age'].mean() );
allAge = allAge.apply(HandleAge)
collections.Counter(survivedAge)
collections.Counter(allAge)

sexProbs = CalculatePercentageOfSurvived( 'Sex', data, None );
nameProbs = CalculatePercentageOfSurvived( 'Name', data, HandleName );
pclassProbs = CalculatePercentageOfSurvived( 'Pclass', data, None );
sipSpProbs = CalculatePercentageOfSurvived( 'SibSp', data, None );
parcProbs = CalculatePercentageOfSurvived('Parch', data, None);
cabinProbs = CalculatePercentageOfSurvived( 'Cabin', data, HandleCabin );
fareProbs = CalculatePercentageOfSurvived( 'Fare', data, HandleFare );
embProbs = CalculatePercentageOfSurvived('Embarked', data, HandleEmbarked);

survivedMale = data.loc[ (data.Survived == True) & (data.Sex=='male')  ]
survivedMale['Age'] = survivedMale['Age'].fillna( data['Age'].median() )
survivedMale[ 'Age' ] = survivedMale['Age'].apply(HandleAge);

survivedMale[ 'Fare' ] = survivedMale['Fare'].apply(HandleFare);
ageHisto = survivedMale[ 'Age' ].hist()
fareHisto = survivedMale[ 'Fare' ].hist()
PlotDictionary(fareProbs)


data.loc[ data.Survived ]['Fare'].hist()


collections.Counter( data['Pclass'] )
collections.Counter( data['SibSp'] )

print(nameProbs)
print(pclassProbs)
print(sipSpProbs)
print(parcProbs)
print(cabinProbs)
print(fareProbs)
print(embProbs)