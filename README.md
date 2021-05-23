![png](img/rm_banner.png?raw=true)

# Table of Contents
* [Abstract](#Abstract)
* [Project Summary](#Project-Summary)
* [Obtain Data](#Obtain-Data)
* [Clean Data](#Clean-Data)
* [Explore Data](#Explore-Data)
* [Model A LinearSVC Model](#Model-A-LinearSVC-Model)
* [Model B Deep Learning with Self Trained Word Embedding](#Model-B-Deep-Learning-with-Self-Trained-Word-Embedding)
* [Model C Deep Learning with GloVe](#Model-C-Deep-Learning-with-GloVe)
* [Model D Model B Plus One More Hidden Layer](#Model-D-Model-B-Plus-One-More-Hidden-Layer)
* [Model E Model C Plus One More Hidden Layer](#Model-E-Model-C-Plus-One-More-Hidden-Layer)
* [Model Interpretation](#Model-Interpretation)
* [Recommendation](#Recommendation)
* [Conclusion](#Conclusion)
* [Future Work](#Future-Work)
* [Dash Applicaiton](#Dash-Applicaiton)
* [Repo Structure](#Repo-Structure)
* [How to Run This Notebook](#How-to-Run-This-Notebook)
<br></br>

# Abstract
The goal of this project is to develop a machine learning model to perform sentiment analysis to find out what our customers love and don't love about our products and services. As a result, we can make recommended improvements.

Following OSMEN framework, we obtain, clean and analyze the data. Perform necessary texts processing before passing the data to our machine learning models. We will develop a few models based on supervised algorithms. These models will be interpreted at the end of the training and validation processes. Model that delivers the best f1-score will be adopted and used to build the Dash application, where the model will be put to test with new unseen data.
<br></br>

# Project Summary
## Business Case
The purpose of this project is to produce a sentiment analysis machine learning model to aid in customer services, market research, social media monitoring and product analytics. The goal in these four areas are the same, that is to leverage machine learning model to identify customer needs and sentiment. As a result, corrective actions can be taken to address the issues.

## Project Approach
The model development process will adhere to the OSEMN framework where data are obtained, cleaned and analyzed.  There after we will create two main types of machine learning models, namely Linear Support Vector Classification and deep learning with Natural Language Processing.  At the end of the training and validation process, we will interpret the results and select the best model based on their f1-score.

## Dash Application
The final model will be used to build the Dash application where users can select a few sets of sample reviews that are unseen by the model during the model development process.  Furthermore, a sandbox area will be provided for users to input their own sample review to test the model accuracy.
<br></br>
    
# Obtain Data    
The data is source from the internet.  You can get a copy from https://nijianmo.github.io/amazon/index.html.  

The Amazon Office Products data set was selected.  It has 306,800 unique products and contains about 5.5 million reviews.  Due to system resource limitations, only about 134,667 reviews are used to train the model.  Furthermore, 5 data sets are subset from the source data and not from the modeling data for testing the model in the Dash application.  

The data set has a review column as well as a review summary column.  It is necessary to combine both columns before we export the data as a CSV file for the next step.  In addition, irrelevant columns are dropped at this stage.
<br></br>

# Clean Data
Firstly, null values are removed.  Secondly, check for reviews that are made up of empty space(s) then they are removed accordingly.  Lastly, create target label of 0 (negative) and 1 (positive) from the "overall" column.  Value of 1 and 2 are mapped to 0, while 4 and 5 are mapped to 1.  Rating  3 or the neutral rating is excluded from the analysis since we are mainly focus on either positive and negative reviews, with emphasis on negative sentiment.
<br></br>

# Explore Data
Before generating the word cloud figure, the Spacy default stop was tweaked because some of these negation words, like no and not, do provide useful information for the model.

![png](img/rm_wrd_cld.png?raw=true)
<br></br>

# Model A LinearSVC Model
1.  Create a new copy of the dataframe.
2.  Clean the review texts by removing:
    <ul>
        <li>HTML tags.</li>
        <li>Stop words.</li>
        <li>Punctuations.</li>
        <li>Digits.</li>
        <li>Single letter.</li>
        <li>Spaces</li>
    </ul>
    The corpus are also lemmatized. 
3. Create train and test data sets.
4. Set up pipeline and grid search parameters.
5. Fit the model and make prediction.  
<br>
<b>Model A Results</b>

![png](img/rm_modelA_results.png?raw=true)

### Observation
1. The grid search process takes about 2.5 hours to run.
2. Both the precision and the recall score for both true positive and true negative predictions are more than satisfactory, with both type of metrics close to 90% and above.
3. It is worth mentioning that the true negative precision and recall scores are quite impressive, given the inherent class imbalance nature of our target.  
<br></br>

# Model B Deep Learning with Self Trained Word Embedding
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

<b>Model Result</b>

![png](img/rm_modelB_results.png?raw=true)

### Observation
1. The precision score for true negative is substantially lower than Model A, .8071 vs .8341.
2. The true positive precision score is very close to Model A, 09748 vs .9815.
3. This model took less than 10 minutes to set up while the Model A took close to 2.5 hrs. Model B's results are more than satisfactory.
<br></br>

# Model C Deep Learning with GloVe
1. Model will use the train and test sets as prepared in model B.
2. Map GloVe vector to X train.
3. Create embedded layer.
4. Use train and validation data as model B.
5. Use the same neural network architecture as model B.  The only difference is the embedding layer, GloVe vs Word2Vec.

![png](img/rm_modelC_seq.png?raw=true)

6. Train the model, evaluate the model with test data.
7. Perform prediction.

<b>Model C Results</b>

![png](img/rm_modelC_results.png?raw=true)

### Observations
1. This model with the pretrained word vectors has the highest true negative precision score of .8432, while Model A and B come in at .8341 and respectively
2. In term of the true positive prediction precision score, it is on par with the other two models.
3. The recall scores from Model B and C are almost identical.
<br></br>

# Model D Model B Plus One More Hidden Layer
This step involves fine tuning model B by introducing one additional layer with difference activation function and half the neurons (50) as the previous hidden layer.

![png](img/rm_modelD_seq.png?raw=true)

<b>Model D Results</b>

![png](img/rm_modelD_results.png?raw=true)

### Observations
1. The precision score for the true negative label increases by 2% over Model B, which has one less hidden layer.
2. A significant drop in the true negative recall score when comparing to Model B, .7425 vs .8213.
<br></br>

# Model E Model C Plus One More Hidden Layer
Same steps taken in model D are applied to this model.

![png](img/rm_modelE_seq.png?raw=true)

Model E Results

![png](img/rm_modelE_results.png?raw=true)

### Observations
1. By far this model produces the highest true negative precision score, .8814. The next higher negative precision score is .8432, which comes from Model C. Both are using the pretrained word vectors.
2. This model also displays the same behavior as observed in Model D, where the true negative recall score drops below 80%, from .8223 to .7745.
<br></br>

### Appendix
Model B and C were tuned with other model training parameters, such as class weight, dropout, flatten layer, RMSprop optimizer. Since they did not improve both models end results, the experiments were not included in this notebook. You can find these tuning experiment in the appendix.ipynb.
<br></br>

# Model Interpretation
1. All the models perform equally well in classifying a true positive, either in precision or recall.
2. Model A has the highest performance scores. However, since the whole process to set up the SVC is substantially longer than the neural network, we will only consider selecting the best among the neural network models. Model C has the highest f1-score, which mean it is performing well in both precision and recall.
3. We will used Model C as our featured model in building the Dash application, which will be use to test new unseen reviews that we set aside at the very begin of this notebook.

![png](img/rm_clsrpt_0.png?raw=true)

![png](img/rm_clsrpt_1.png?raw=true)
<br></br>

# Recommendation
1. Customer Services.  Sentiment analysis can help us identify specific issue raised by the customers, either through emails or live feedback.
2. Market Research.  In market research, we can find out how and what our competitors are doing.  What people like about their products.  Can we include new features in our product line to win the customers over.
3. Social Media Monitoring.  We know people like to tweet or post what they love or hate a product.  With machine learning, we can wade through massive unstructured data to find out what they commend about our products and services.  Therefore, negative concerns can be addressed quickly.
4. Product Analytics.  An example of product analytics would be to monitor customers attitude toward the launching of a new product.
<br></br>

# Conclusion
In this project, we explore two main types of machine learning models, linear support vector classification and deep learning with natural language processing, to classify positive and negative reviews.

Although there are other type of modeling in supervised branch, such as logistic regression and random forest. We went with the Support Vector Machines (SVC) model for no particular reason, other than needing a base model to compare the results from the neural network model.

Through the help of grid search, we were able to develop a decent SVC model that yields a f1-score of 0.851 for predicting the negative label and a f1-score of 0.978 for predicting the positive label. This model produces the highest positive prediction precision score among all the models. However, we have to keep in mind that the model took about 2.5 hours to train.

In the deep learning models area, we endeavored to train our own word vectors and compare the results to another model that uses a pretrained word vector from Stanford, called GloVe. It is evident that Model C with the pretrained word vector perform very well over our self-trained word vectors Model B, with the true negative f1-score of .835 vs .814.

Next we fine tuned both Model B and C by introducing an additional hidden layer with 50 neurons. Both model D and E show a decrease in the true negative f1-score, compared to Model C. One the other hand, the true positive f1-score improved marginally.

Model C will be deployed in our Dash application, where new unseen reviews will be put through the test to how well our model performs.
<br></br>

# Future Work
1. Try out other supervised learning models, such as Logistic Regression or Random Forest. To see how well these model will measure up to Support Vector Machine model.
2. Employ alternate pretrained word vectors, such as Google's Word2Vec. Word2Vec and GloVe are trained based on different technique. The former uses shallow feedforward neural network while the latter uses matrix factorization. Thus it is worth exploring Word2Vec and see how it can improve our nueral network model.
3. Increase the word vectors dimension from 100 to 150 and 200 as more dimension can store more information.
4. Add more hidden layer to the network since we did see an improvement to the precision score once we add a second hidden layer.
<br></br>

# Dash Applicaiton
The Dash application is built with Jupyter Dash and currently only runs locally.  Alternatively, you may run with app with the command "python app.py" at a console.  The app.py is built with Dash, as such you don't need to run this notebook in order to run the Dash application.

The at "Obtain Data" stage, we reserved 5 set of data.  They are created based on the following rules:
<ul>
    <li>Each set contains one office product.  Each successive set will increment by no more than 500 review counts than the previous set.  For instance, set 1 will have 500 reviews or more but less than 1000.  The second set will have 1000 review or more but less than 1500.</li>
    <li>The negative review proportion to total review in the set should fall within 10% to 15%.</li>
</ul>
<br></br>

Screenshot of the Dash application when first loaded.

![png](img/rm_dash_tab1.png?raw=true)

In the "Model Showcase" tab, sample product reviews are provided.  These are data that the model have not been trained or tested on before.  Users can select the provided data set from the drop down menu.

The 'Get Prediction' button, allows user to fit the data to the model.  The prediction results will be appended to the selected data set and displayed in the 'Actual vs Prediction' pane.  The precision score of the model prediction will be plotted in the 'Precision Score' Pane.
<br></br>

![png](img/rm_dash_tab2.png?raw=true)

In the "Sand box" tab, users can type in their own review or copy and paste from external sources to get a prediction from the model.

Below the 'Single Prediction' section, users can upload csv file and let the model generates the labels/predictions.  Results will be shown under the 'Upload File Prediction Result' pane.  If a label is provided in the upload file, the precision score will be charted as well.

# Repo Structure
1. Root of the repo contains the main notebook, a customed function py file and a PowerPoint about this project.
2. The data folder contains source data file, subset of the data file and the Stanford GloVe word embdedding file.
3. The dump foloder contains saved models and variables.  
4. The img folder contains jpg and png for this readme.
<br></br>
# How to Run This Notebook
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
<br></br>