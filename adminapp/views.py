from django.shortcuts import render,redirect
from adminapp.models import *
from django.contrib import messages
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score,r2_score
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from adminapp.models import * 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from userapp.models import *
from mainapp.models import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim  
from sklearn.svm import SVR
import numpy as np
from .models import GAN  # Make sure to import your GAN model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .models import RNN
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


# Create your views here.
def admin_index(request):
    dataset=Dataset.objects.all().count()
    user=UserModel.objects.all().count()
    test=TestingModel.objects.all().count()
    return render(request,'admin/admin-index.html',{'Dataset':dataset,'user':user,'test':test})

def admin_uploaddata(request):
    if request.method == 'POST' :
        dataset = request.FILES['dataset']
        data = Dataset.objects.create(data_set = dataset)
        data = data.data_id
        print(type(data),'type')


        return redirect('admin_run_algorithms')
    return render(request,'admin/admin-uploaddata.html')
def GAN_alg(request):
    return render(request,"admin/GAN_btn.html")
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from django.shortcuts import render
from django.contrib import messages
from .models import GAN  # Import your GAN model

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output for price
        )

    def forward(self, x):
        return self.model(x)

# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from django.contrib import messages
from django.shortcuts import render
from .models import GAN  # Make sure to import your GAN model

# Define your Generator and Discriminator models here...

# Function to train the GAN
def train_gan(generator, discriminator, num_samples, noise_dim, y_min, y_max, num_epochs=10):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Number of samples for this iteration
        batch_size = 100  # You can adjust this as needed

        # Real data labels
        real_labels = torch.ones(batch_size, 1)  # Labels for real data
        real_data = torch.randn(batch_size, 1) * (y_max - y_min) + y_min  # Replace with your actual normalized data
        outputs = discriminator(real_data)
        d_loss_real = criterion(outputs, real_labels)
        
        # Generate fake data
        noise = torch.randn(batch_size, noise_dim)
        fake_data = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)  # Labels for fake data
        outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        # Backpropagation and optimization for Discriminator
        optimizer_disc.zero_grad()
        (d_loss_real + d_loss_fake).backward()
        optimizer_disc.step()

        # Train Generator
        optimizer_gen.zero_grad()
        outputs = discriminator(fake_data)
        g_loss = criterion(outputs, real_labels)  # We want to fool the discriminator
        g_loss.backward()
        optimizer_gen.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss_real.item() + d_loss_fake.item():.4f}, G Loss: {g_loss.item():.4f}')

# View function for GAN
def GAN_btn(request):
    df = pd.read_csv('DATASET/Clean_Dataset.csv')

    # Preprocessing
    X = df[['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left']]
    y = df['price'].values

    # Normalize target variable
    y_min = y.min()
    y_max = y.max()
    y_normalized = (y - y_min) / (y_max - y_min)

    label_encoder = LabelEncoder()
    for col in ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
        X[col] = label_encoder.fit_transform(X[col])

    scaler = StandardScaler()
    X[['duration', 'days_left']] = scaler.fit_transform(X[['duration', 'days_left']])

    # Train-Test split
    X_train, _, y_train, _ = train_test_split(X, y_normalized, test_size=0.2, random_state=1)

    input_dim = X_train.shape[1]
    noise_dim = 10  # You can adjust the noise dimension

    generator = Generator(input_dim=noise_dim)
    discriminator = Discriminator(input_dim=1)  # Discriminator input size for price

    # Train the GAN model
    train_gan(generator, discriminator, len(X_train), noise_dim, y_min, y_max, num_epochs=100)

    # Generate new prices
    noise = torch.randn(100, noise_dim)  # Generate noise for 100 samples
    generated_data = generator(noise).detach().numpy()
    for price in generated_data:
        GAN.objects.create(Generated_Data=price[0], Name="GAN Generated Price")

    # Fetch the last stored metrics for display
    data = GAN.objects.last()

    messages.success(request, 'GAN Algorithm executed successfully!')
    return render(request, 'admin/GAN_btn.html', {'data': data})











def RNN_alg(request):
    return render(request,"admin/RNN_btn.html") 






# Define the RNN Model



# Training function for RNN
def train_model(model, X_train_tensor, y_train_tensor, epochs=10):
    criterion = nn.MSELoss()  # Use MSE Loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Calculate loss using MSE
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')  # Print loss

# Evaluation function for RNN
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)  # Output layer for regression (single continuous value)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out  # Direct output for regression



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        y_pred_tensor = model(X_test_tensor)  # Get predictions
        y_pred = y_pred_tensor.numpy()  # Convert predictions to numpy array
        y_test = y_test_tensor.numpy()  # Convert actual values to numpy array

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, mae, r2
# RNN view function

def RNN_btn(request):
    df = pd.read_csv('DATASET/Clean_Dataset.csv')

    # Preprocessing
    X = df[['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left']]
    y = df['price'].values

    # Normalize target variable to be between 0 and 1 if required for specific application
    y_min = y.min()
    y_max = y.max()
    y_normalized = (y - y_min) / (y_max - y_min)  # Normalize price

    label_encoder = LabelEncoder()
    for col in ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
        X[col] = label_encoder.fit_transform(X[col])

    scaler = StandardScaler()
    X[['duration', 'days_left']] = scaler.fit_transform(X[['duration', 'days_left']])

    X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=1)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    seq_length = 1
    input_dim = X_train.shape[1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, seq_length, input_dim)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, seq_length, input_dim)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = RNNModel(input_dim=input_dim)

    # Train the RNN model
    train_model(model, X_train_tensor, y_train_tensor, epochs=10)

    # Evaluate the model using regression metrics
    mse, mae, r2 = evaluate_model(model, X_test_tensor, y_test_tensor)

    # Save metrics to the database
    RNN.objects.create(
        Mean_Squared_Error=round(mse, 2),
        Mean_Absolute_Error=round(mae, 2),
        R2_Score=round(r2, 2),
        Name="RNN Regression Model"
    )

    # Fetch the last stored metrics for display
    data = RNN.objects.last()

    messages.success(request, 'RNN Algorithm executed successfully!')
    return render(request, 'admin/RNN_btn.html', {'i': data})


# def admin_run_algorithms(request):
#     data = Dataset.objects.all().order_by('-data_id').first()
    
#     print(data,type(data),'sssss')
#     file = str(data.data_set)
#     df = pd.read_csv('./media/'+ file)
#     table = df.to_html(table_id='data_table')
#     # print(df.iloc[0,[1]])
#     # print(len(df))
#     # for i in range(len(df)):
#     #     print(df.iloc[i,[1]],'loop')
#     # print(df[0:5],type(df),'database tablesssssssssss')

#     return render(request,'admin/admin-run-algorithms.html',{'i':data,'t':table})



 


# def score(request,id):
#     data = Dataset.objects.get(data_id=id)

#     return render(request,'admin/admin-score.html',{'i':data})




def admin_sentiment(request):
    try:    
        data = Dataset.objects.all().order_by('-data_id').first()
        dt_ac = data.dt_Accuracy*100
    
        sv_ac = data.svr_Accuracy*100
        
        nb_ac = data.knn_Accuracy*100

        lr_ac = data.lr_Accuracy*100

        rf_ac = data.rf_Accuracy*100

        print(rf_ac,lr_ac,nb_ac,sv_ac,dt_ac)


    
        context = {
            'lr_ac':lr_ac,
            
            'nb_ac':nb_ac,
            
            'dt_ac':dt_ac,

            'rf_ac':rf_ac,

            'sv_ac':sv_ac,
            
        }
        return render(request,'admin/admin-sentiment-analysis.html',context)
    except:
        messages.info(request,'Run all 4 algorithms')

        return redirect('admin_run_algorithms')
    






# def RandomForest(request,id):
#     Accuracy = None
#     data = Dataset.objects.get(data_id=id)
#     id = data.data_id
#     file = str(data.data_set)
#     df = pd.read_csv('./media/'+ file)
#     X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
#           'Cochin','Hyderabad','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min']]
#     y=df['Price']
#     print(y.head(),'gggggggggggggggggggggggggggggggggggg')
#     from sklearn.model_selection import train_test_split
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#     from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
#     from sklearn.metrics import accuracy_score,confusion_matrix
#     def prediction(ml_model):
#         print('Model is: {}'.format(ml_model))
#         model= ml_model.fit(X_train,y_train)
#         print("Training score: {}".format(model.score(X_train,y_train)))
#         predictions = model.predict(X_test)
#         print("Predictions are: {}".format(predictions))
#         print('\n')
#         Accuracy=r2_score(y_test,predictions) 
#         print(Accuracy,'ssssssssssssssssssssssssss')
#         print("r2 score is: {}".format(Accuracy))
#         print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
#         print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
#         print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
#         from sklearn.model_selection import RandomizedSearchCV
#         random_grid = {
#         'n_estimators' : [100, 120, 150, 180, 200,220],
#         'max_features':['auto','sqrt'],
#        'max_depth':[5,10,15,20], }
#         rf=RandomForestRegressor()
#         rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1,)

#         rf_random.fit(X_train,y_train)

# # best parameter
#         rf_random.best_params_
#         prediction = rf_random.predict(X_test)
#         Accuracy2=r2_score(y_test,prediction)
#         data.rf_Accuracy=Accuracy2
#         data.rf_algo = "Random Forest"
#         data.save()
#         import joblib
#         file=open('airline_rf.pkl','wb')
#         joblib.dump(rf_random,file)
#     from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
#     prediction(RandomForestRegressor())
#     # import pickle
#     # file=open('airline_rf.pkl','wb')
#     # pickle.dump(model,file)
#     return redirect('score',id=id)

# def button(request,id):
#     import pickle
   
#     test=TestingModel.objects.get(pk=id)
#     print(test,'jjjjjjjjjjjjjjjjjjjjjjjj')
#     X_test= [[test.Total_Stops,test.Air_India,test.GoAir,test.IndiGo,test.Jet_Airways,test.Jet_Airways_Business
#     ,test.Multiple_carriers,test.Multiple_carriers_Premium_economy,test.SpiceJet,test.Trujet,test.Vistara,test.Vistara_Premium_economy,
#     test.Chennai,test.Delhi,test.Kolkata,test.Mumbai,test.Cochin,test.Hyderabad,test.journey_day,test.journey_month,
#     test.Dep_Time_hour,test.Dep_Time_min,test.Arrival_Time_hour,test.Arrival_Time_min,test.dur_hour,test.dur_min]]
    
#     print(X_test,'iiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
#     import joblib
#     import pickle
#     model=open('airline_rf.pkl','rb')
#     rf_random=joblib.load(model)
#     # from sklearn.ensemble import RandomForestRegressor
#     y_pred=rf_random.predict(X_test)
#     # y_prediction=forest.predict(data1)
#     # Accuracy=metrics.r2_score(y_test,y_prediction)
#     print(y_pred,'uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
#     messages.info(request,y_pred[0])
#     # messages.warning(request,Accuracy)
#     messages.success(request,'Predicted Successfully')
#     return redirect('user_index')
#     # return redirect('score',id=33)
 
#     # try:
       
#     #      print(id,'iiiiiiiiiidddddddddddd')
#     #     test=TestModel.objects.get(id=id)
            
#     #     test=TestModel.objects.get(pk=id)
#     #     print(test,'jjjjjjjjjjjjjjjjjjjjjjjj')
#     #     data1= [[test.airline,test.source,test.to,test.daysleft_travel,test.dept_time,test.arr_time,test.class1,test.stops]]
#     #     y_test=reg_rf.predict(data1)
#     #     print(y_test,'yyyyyy')
#     #     messages.info(request,y_test[0])

#     #     messages.warning(request,Accuracy)
#     #     messages.success(request,'Predicted Successfully') 

#     #     return redirect('user_index')

    
#     # except:
#     #         pass return redirect('score',id=33)

# def DecisionTree(request,id):
#     Accuracy = None
#     data = Dataset.objects.get(data_id=id)
#     id = data.data_id
#     file = str(data.data_set)
#     df = pd.read_csv('./media/'+ file)
#     X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
#           'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min']]
#     y=df['Price']
#     print(y.head(),'gggggggggggggggggggggggggggggggggggg')
#     from sklearn.model_selection import train_test_split
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#     from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
#     from sklearn.metrics import accuracy_score,confusion_matrix
#     def prediction(ml_model):
#         print('Model is: {}'.format(ml_model))
#         model= ml_model.fit(X_train,y_train)
#         print("Training score: {}".format(model.score(X_train,y_train)))
#         predictions = model.predict(X_test)
#         print("Predictions are: {}".format(predictions))
#         print('\n')
#         Accuracy=r2_score(y_test,predictions) 
#         print(Accuracy,'ssssssssssssssssssssssssss')
#         print("r2 score is: {}".format(Accuracy))
#         print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
#         print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
#         print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
#         data.dt_Accuracy=Accuracy
#         data.dt_algo = "DecisionTree"
#         data.save()
#     from sklearn.tree import DecisionTreeRegressor
#     prediction(DecisionTreeRegressor())
   
#     return redirect('score',id=id)
   
#     return redirect('score',id=id)

# def KNeighborsRegressor(request,id):
#     Accuracy = None
#     data = Dataset.objects.get(data_id=id)
#     id = data.data_id
#     file = str(data.data_set)
#     df = pd.read_csv('./media/'+ file)
#     X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
#           'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min'
# ]]
#     y=df['Price']
#     print(y.head(),'gggggggggggggggggggggggggggggggggggg')
#     from sklearn.model_selection import train_test_split
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#     from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
#     from sklearn.metrics import accuracy_score,confusion_matrix
#     def prediction(ml_model):
#         print('Model is: {}'.format(ml_model))
#         model= ml_model.fit(X_train,y_train)
#         print("Training score: {}".format(model.score(X_train,y_train)))
#         predictions = model.predict(X_test)
#         print("Predictions are: {}".format(predictions))
#         print('\n')
#         Accuracy=r2_score(y_test,predictions) 
#         print(Accuracy,'ssssssssssssssssssssssssss')
#         print("r2 score is: {}".format(Accuracy))
#         print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
#         print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
#         print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
#         data.knn_Accuracy=Accuracy
#         data.knn_algo = "KNNeighbor"
#         data.save()
#     from sklearn.neighbors import KNeighborsRegressor
#     prediction(KNeighborsRegressor())
   
#     return redirect('score',id=id)

# def LinearRegressor(request,id):
#     Accuracy = None
#     data = Dataset.objects.get(data_id=id)
#     id = data.data_id
#     file = str(data.data_set)
#     df = pd.read_csv('./media/'+ file)
#     X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
#           'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min'
# ]]
#     y=df['Price']
#     print(y.head(),'gggggggggggggggggggggggggggggggggggg')
#     from sklearn.model_selection import train_test_split
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#     from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
#     from sklearn.metrics import accuracy_score,confusion_matrix
#     def prediction(ml_model):
#         print('Model is: {}'.format(ml_model))
#         model= ml_model.fit(X_train,y_train)
#         print("Training score: {}".format(model.score(X_train,y_train)))
#         predictions = model.predict(X_test)
#         print("Predictions are: {}".format(predictions))
#         print('\n')
#         Accuracy=r2_score(y_test,predictions) 
#         print(Accuracy,'ssssssssssssssssssssssssss')
#         print("r2 score is: {}".format(Accuracy))
#         print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
#         print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
#         print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
#         data.lr_Accuracy=Accuracy
#         data.lr_algo = "Linear Regressor"
#         data.save()
#     from sklearn.linear_model import LogisticRegression
#     prediction(LogisticRegression())
#     return redirect('score',id=id)

# def SVR(request,id):
#     Accuracy = None
#     data = Dataset.objects.get(data_id=id)
#     id = data.data_id
#     file = str(data.data_set)
#     df = pd.read_csv('./media/'+ file)
#     X=df[['Total_Stops','Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','Multiple carriers','Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy','Chennai','Delhi','Kolkata','Mumbai',
#           'Cochin','Delhi','Hyderabad','Kolkata','journey_day','journey_month','Dep_Time_hour','Dep_Time_min','Arrival_Time_hour','Arrival_Time_min','dur_hour','dur_min'
# ]]
#     y=df['Price']
#     print(y.head(),'gggggggggggggggggggggggggggggggggggg')
#     from sklearn.model_selection import train_test_split
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#     from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
#     from sklearn.metrics import accuracy_score,confusion_matrix
#     def prediction(ml_model):
#         print('Model is: {}'.format(ml_model))
#         model= ml_model.fit(X_train,y_train)
#         print("Training score: {}".format(model.score(X_train,y_train)))
#         predictions = model.predict(X_test)
#         print("Predictions are: {}".format(predictions))
#         print('\n')
#         Accuracy=r2_score(y_test,predictions) 
#         print(Accuracy,'ssssssssssssssssssssssssss')
#         print("r2 score is: {}".format(Accuracy))
#         print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
#         print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
#         print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
#         data.svr_Accuracy=Accuracy
#         data.svr_algo = "SVR"
#         data.save()
#     from sklearn.svm import SVR
#     prediction(SVR())
#     return redirect('score',id=id)


