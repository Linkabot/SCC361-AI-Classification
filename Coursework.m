%                          Data Preperation 

% Reading the Text Emotion Data file.
fullFilteredData = readtable('text_emotion_data_filtered.csv');
% Splitting the data to only have the Content colmun.
textData = split(fullFilteredData.Content, newline);
% Tokenizing every tweet to be stored in the Bag of Words.
documents = tokenizedDocument(textData);
% Creating the Bag of Words with the Tokenized tweets. 
bag = bagOfWords(documents);
% Removing all Stopwords and all Infrequest words with less than 100 
% occurrences from the bag of words and storing it in a new bag.
newBag = removeWords(bag, stopWords);
newBag = removeInfrequentWords(newBag, 100);
% Build the Term Frequency-Inverse Document Frequency matrix with the
% newBag.
M1 = tfidf(newBag);
% Build a corresponding label vector from the column of sentiments.
labels = table2array(fullFilteredData(:,1));

%                         Features and Labels

% Create training and testing features with tf-idf. Also create
% corresponding training and testing labels. 
trainingFeature = full(M1(1:6432, :)); 
trainingLabels = labels(1:6432, :); 
testingFeature = full(M1(6433:end, :));
testingLabels = labels(6433:end, :);

%                     Model Training and Evaluation 

%     K-Nearest Neighbour model

% Syntax to train machine learning model for the K-Nearest Neighbour model. 
kNear = fitcknn(trainingFeature, trainingLabels);
predictKnear = predict(kNear, testingFeature);
% Confusion chart using testing labels and prediction from K-Nearest
% Neighbour model.
kNConfuse = confusionchart(testingLabels, predictKnear);
% Equation to find the Accuracy of the predictions. 
NormalKN = sum(kNConfuse.NormalizedValues, 'all');
CorrectNormalKN = sum(diag(kNConfuse.NormalizedValues), 'all');
kNAccuracy = CorrectNormalKN / NormalKN

%     Naive Bayes Model

% Syntax to train machine learning model for the Naive Bayes model.
nBayes = fitcnb(trainingFeature, trainingLabels);
predictnBayes = predict(nBayes, testingFeature);
% Confusion chart using testing labels and prediction from Naive Bayes
% model.
nBConfuse = confusionchart(testingLabels, predictnBayes);
% Equation to find the Accuracy of the predictions.
NormalNB = sum(nBConfuse.NormalizedValues, 'all');
CorrectNormalNB = sum(diag(nBConfuse.NormalizedValues), 'all');
nBAccuracy = CorrectNormalNB / NormalNB

%     Discriminant Analysis Model

% Syntax to train machine learning model for the Discriminant Analysis model.
discrimAnalysis = fitcdiscr(trainingFeature, trainingLabels);
predictnDa = predict(discrimAnalysis, testingFeature);
% Confusion chart using testing labels and prediction from Discriminant
% Analysis model.
dAConfuse = confusionchart(testingLabels, predictnDa);
% Equation to find the Accuracy of the predictions.
NormalDA = sum(dAConfuse.NormalizedValues, 'all');
CorrectNormalDA = sum(diag(dAConfuse.NormalizedValues), 'all');
dAAccuracy = CorrectNormalDA / NormalDA
