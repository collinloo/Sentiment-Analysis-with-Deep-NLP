## Table of Contents
* [Repo Structure](#Repo-Structure)
* [How to Run This Notebook](#How-to-Run-This-Notebook)
* [Project Summary](#Project-Summary)
* [Obtain Data](#Obtain-Data)
* [Clean Data](#Clean-Data)
* [Explore Data](#Explore-Data)
* [Model A Linear SVM Model](#Model-A-Linear-SVM-Model)
* [Model B Deep Learning with Self Trained Word Embedding](#Model-B-Deep-Learning-with-Self-Trained-Word-Embedding)
* [Model C Deep Learning with GloVe](#Model-C-Deep-Learning-with-GloVe)
* [Model D: Model B Plus One More Hidden Layer](#Model-D:-Model-B-Plus-One-More-Hidden-Layer)
* [Model E: Model C Plus One More Hidden Layer](#Model-E:-Model-C-Plus-One-More-Hidden-Layer)
* [Model Interpretation](#Model-Interpretation)
* [Conclusion](#Conclusion)
* [Future Work](#Future-Work)
* [Dash Applicaiton](#Dash-Applicaiton)

## Repo Structure
1. Root of the repo contains the main notebook, a customed function py file and a PowerPoint about this project.
2. The data folder contains source data file, subset of the data file and the Stanford GloVe word embdedding file.
3. The dump foloder contains saved models and variables.  
4. The img folder contains jpg and png for this readme.

## How to Run This Notebook
1. Create a local repo by cloning this repo
2. Required pacakges:
<ul>
    <li>dash</li>
    <li>json</li>
    <li>gensim</li>
    <li>nltk</li>
    <li>numpy</li>
    <li>pandas</li>
    <li>ploty</li>
    <li>python > 3.7</li>
    <li>sklearn</li>
    <li>spacy</li>
    <li>tensorflow</li>
</ul>
3. To execute the cells in the notebook with the saved data, proceed to section 5 'Clean Data' and select from the top menu 'Cell > Run All Below' command.
    
## Project Summary
### Business Case
The purpose of this project is to produce a machine learning sentiment analysis model to aid in product analysis.  With the machine learning model, we can identify customer needs  and sentiment.  In addition, competitors analysis can be performed on similar products or services.  As a result, recommendations can be made accordingly.

### Project Approach
The model development process will adhere to the OSEMN framework where data are obtained, cleaned and analyzed.  There after we will create two mian types of machine learning models, namely supervised and unsupervised.  At the end of the training and validation process, we will interpret the results and select the best model based on their precision scores.

### Dash Application
The final model will be used to build the Dash application where users can select a few sets of sample reviews that are unseen by the model during the model development process.  Furthermore, a sandbox area will be provided for users to input their own sample review to test the model accuracy  
    
## Obtain Data    
The data is source from the internet.  You can get a copy from https://nijianmo.github.io/amazon/index.html.  

The Amazon Office Products data set was selected.  It has 306,800 unique products and contains about 5.5 million reviews.  Due to system resource limitations, only about 134,667 reviews are used to train the model.  Furthermore, 5 data sets are subset from the source data and not from the modeling data for testing the model in the Dash application.  

The data set has a review column as well as a review summary column.  It is necessary to combine both columns before we export the data as a CSV file for the next step.  In addition, irrelevant columns are dropped at this stage.

## Clean Data
Firstly, null values are removed.  Secondly, check for reviews that are made up of empty space(s) then they are removed accordingly.  Lastly, create target label of 0 (negative) and 1 (positive) from the "overall" column.  Value of 1 and 2 are mapped to 0, while 4 and 5 are mapped to 1.  Rating  3 or the neutral rating is excluded from the analysis since we are mainly focus on either positive and negative reviews, with emphasis on negative sentiment.

## Explore Data
Before generating the word cloud figure, the Spacy default stop was tweaked because some of these negation words, like no and not, do provide useful information for the model.

![png](img/rm_wrd_cld.png?raw=true)

## Model A Linear SVM Model
1.  Create a new copy of the dataframe.
2.  Clean the review texts by removing:
<ul>
    <li>HTML tags.</li>
    <li>Stop words.</li>
    <li>Punctuations.</li>
    <li>Digits.</li>
    <li>Single letter.</li>
    <li>Spaces</li>
</ul> The corpus are also lemmatized. 
3. Create train and test data sets.
4. Set up pipeline and grid search parameters.
5. Fit the model and make prediction.

![png](img/rm_modelA_results.png?raw=true)

The grid search process takes about 2.5 hours to run.  The precision score for positive prediction is about 0.98.  As for the precision score for negative prediction, the model guesses it right  about 83% at a time.  Given the class imbalance in the data set, which is about 12% of the testing data, the result is more than adequate.

## Model B Deep Learning with Self Trained Word Embedding

1. Create X and y.
2. Clean X the same way we do for Model A.
3. Tokenize X.
4. Split X into train and test sets.
5. Train word vector with Word2Vec.
6. Further divide training into train and validation.
7. Create the embedded layer.
8. Setup neural network.  

![png](img/rm_modelB_seq.png)

9. Train the model, evaluate the model with test data.
10. Perform prediction.

![png](img/rm_modelB_results.png?raw=true)

The precision score for negative prediction is much lower than model A, .807 vs .834.  However, this model only takes less than 10 minutes to train.  Both model precision scores are very close to each other.

## Model C Deep Learning with GloVe
1. Model will use the train and test sets as prepared in model B.
2. Map GloVe vector to X train.
3. Create embedded layer.
4. Use train and validation data as model B.
5. Use the same neural network architecture as model B.  The only difference is the embedding layer, GloVe vs Word2Vec.

![png](img/rm_modelC_seq.png?raw=true)

6. Train the model, evaluate the model with test data.
7. Perform prediction.

![png](img/rm_modelC_results.png?raw=true)

The precision score for negative prediction is higher by 4% as compared to model B, .848 vs .807.  The precision score for positive prediction are almost identical for both model B and C.

## Model D: Model B Plus One More Hidden Layer
This step involves fine tuning model B by introducing one additional layer with difference activation function and half the neurons (50) as the previous hidden layer.

![png](img/rm_modelD_seq.png?raw=true)

Model performance results
![png](img/rm_modelD_results.png?raw=true)

By adding one more hidden layer with 50 neurons, the model precision score for negative prediction improved by 2%, from .807 to .820.  On the flip side, precision score for positive prediction dropped by 1%, from .974 to .964.

## Model E: Model C Plus One More Hidden Layer
Same steps taken in model D are applied to this model.

![png](img/rm_modelE_seq.png?raw=true)

Model performance results
![png](img/rm_modelE_results.png?raw=true)

Model E, by far, produces the best precision score for negative prediction among all the models we have developed.  It can correctly label a true negative 88% at a time while correctly label a true positive at 97% at a time.  For a model that takes less than 10 minutes to train, the result is more than satisfactory.

## Model Interpretation
1. All the models perform equally well in classifying a review correctly, that is when it is a positive review it will not label it as a negative review and vise versa.
2. Model E is considered more superior because of its high precision in predicting true negative label, which is our main focus.
3. We will used Model E as our featured model in building the Dash application, which will be use to test new unseen reviews that we set aside at the very begin of this notebook.

![png](img/rm_neg_compare.png?raw=true)

![png](img/rm_pos_compare.png?raw=true)

## Conclusion
In this project, we explore two main types of machine learning models, supervised and unsupervised models, also known as deep learning, to classify positive and negative reviews.

Model A belongs to the supervised branch. Although there are other type of modeling in supervised branch, such as logistic regression and random forest. We went with the Support Vector Machines (SVM) model for no particular reason, other than needing a base model to compare the results from the unsupervised models.

Through the help of grid search, we were able to develop a decent SVM model that yields a precision score of 0.8341 for predicting the negative label and a precision score of 0.9815 for predicting the positive label. This model produces the highest positive prediction precision score among all the models. However, we have to keep in mind that the model took about 2.5 hours to train.

In the deep learning models area, we endeavored to train our own word vectors and compare the results to another model that uses a pretrained word vector from Stanford, called GloVe. With the main focus on the negative label precision, it is evident that Model C with the pretrained word vector perform very well over our self-trained word vectors Model B (0.8486 vs 0.8071 for negative label precision score).

Next we fine tuned both Model B and C by introducing an additional hidden layer with 50 neurons. We see significant improvements in both models, regardless the type of word vectors used in the embedding layers. Model D (Model B with additional layer) yields a true negative precision score of .8202. Model E (Model C with additional layer) yields a true negative precision score of .8814. The difference in the true positive precision from both models are negligible.

Model E will be deployed in our Dash application, where new unseen reviews will be put through the test to how well our model performances.

## Future Work
1. Try out other supervised learning models.
2. Employ alternate pretrained word vectors, such as Google's Word2Vec.
3. Increase the word vectors dimension from 100 to 150 and 200.
4. Add more hidden layers to the network since we did see an improvement to the precision score once we add a second hidden layer.


## Dash Applicaiton
The Dash application is built with Jupyter Dash and currently only runs locally.

The at "Obtain Data" stage, we reserved 5 set of data.  They are created based on the following rules:
<ul>
    <li>Each set contains one office product.  Each successive set will increment by no more than 500 review counts than the previous set.  For instance, set 1 will have 500 reviews or more but less than 1000.  The second set will have 1000 review or more but less than 1500.</li>
    <li>The negative review proportion to total review in the set should fall within 10% to 15%.</li>
</ul>

These data set can be selected from a dropdown box.  Once user select a set of data, the csv will be loaded into a pandas dataframe in the background and then convert to Dash table.  At the same time the word cloud figures for the data set will be displayed.  To save the page loading time, all the word cloud figures are pre-generated.

Screenshot of the Dash application when first loaded.

![png](img/rm_dash.png?raw=true)

Under the "Model Summary" Section, clicking the get prediction button will perform the model.predict on the selected data set in the background.  Upon completion, the page will be updated with classification report and the confusion matrix plot.

At the bottom of the page is where user can type in their own texts to test the model.  Click the try me button, will perform the prediction on the input text.

![png](img/rm_single_pred.png?raw=true)