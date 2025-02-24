from django.shortcuts import render,redirect, HttpResponse
from mainapp.models import *
from django.contrib import messages
from userapp.models import *
from adminapp.models import *
import pandas as pd

from django.http import Http404


import numpy as np
import torch
from django.http import Http404
from django.shortcuts import render
from datetime import datetime

from .gan import Generator, Discriminator  # Assuming these are your GAN model classes

import numpy as np
import torch
from django.http import Http404
from django.shortcuts import render
from .gan import Generator, Discriminator  # Assuming these are your GAN model classes

def Predict(request, flight_id):
    try:
        # Fetch the flight data using flight_id from the dataset
        flight = Flight.objects.get(id=flight_id)
    except Flight.DoesNotExist:
        raise Http404("Flight not found")

    # Extract input features from the dataset fields
    total_stops = flight.stops
    airline_features = [
        flight.Air_India, flight.GoAir, flight.IndiGo, flight.Jet_Airways, 
        flight.Jet_Airways_Business, flight.Multiple_carriers, 
        flight.Multiple_carriers_Premium_economy, flight.SpiceJet, flight.Trujet
    ]
    
    # One-hot encode the source and destination cities
    source_city = [flight.Bangalore, flight.Hyderabad, flight.Kolkata, flight.Delhi, flight.Cochin]
    destination_city = [flight.Bangalore, flight.Hyderabad, flight.Kolkata, flight.Delhi, flight.Cochin]
    
    # Time-related features
    dep_hour = flight.dept_time.hour
    dep_minute = flight.dept_time.minute
    arr_hour = flight.arr_time.hour
    arr_minute = flight.arr_time.minute
    
    # Duration (hours and minutes)
    duration_hour = (flight.arr_time - flight.dept_time).seconds // 3600
    duration_minute = ((flight.arr_time - flight.dept_time).seconds // 60) % 60
    
    # Journey date details
    journey_day = flight.dept_time.day
    journey_month = flight.dept_time.month

    # Prepare the GAN input array for price prediction
    gan_input = np.array([
        total_stops,
        dep_hour,
        dep_minute,
        arr_hour,
        arr_minute,
        duration_hour,
        duration_minute,
        journey_day,
        journey_month,
        *airline_features,
        *source_city,
        *destination_city
    ])

    # Define input and output dimensions for the Generator
    input_dim = len(gan_input)
    output_dim = 1  # The output will be the predicted price

    # Load the generator model for price prediction
    generator = Generator(input_dim=input_dim, output_dim=output_dim)
    generator.load_state_dict(torch.load('generator.pth'), strict=False)
    generator.eval()

    # Pass the input through the generator to get the predicted price
    with torch.no_grad():
        gan_output = generator(torch.tensor(gan_input, dtype=torch.float32).view(1, -1))

    # Load the discriminator model to validate the generated price (optional)
    discriminator = Discriminator(input_dim=output_dim)
    discriminator.load_state_dict(torch.load('discriminator.pth'), strict=False)
    discriminator.eval()

    # Validate the generated price through the discriminator (optional)
    with torch.no_grad():
        gan_prediction = discriminator(gan_output)

    # Get the predicted price
    prediction_value = gan_output.item()

    # Retrieve the last predicted price from session (if exists)
    last_prediction = request.session.get('last_prediction', None)

    # Determine if the price has increased or decreased
    if last_prediction is not None:
        if prediction_value > last_prediction:
            price_change_message = "The predicted price has increased."
        elif prediction_value < last_prediction:
            price_change_message = "The predicted price has decreased."
        else:
            price_change_message = "The predicted price remains the same."
    else:
        price_change_message = "This is the first prediction."

    # Store the current prediction in the session
    request.session['last_prediction'] = prediction_value

    # Pass the prediction and messages to the template
    return render(request, 'user/user-index.html', {
        'prediction': prediction_value,  # Predicted price
        'price_change_message': price_change_message,  # Message about price change
        'total_stops': total_stops,       # Display number of stops
        'dep_time': f"{dep_hour}:{dep_minute:02d}",  # Display formatted departure time
        'arr_time': f"{arr_hour}:{arr_minute:02d}",  # Display formatted arrival time
        'duration': f"{duration_hour}h {duration_minute}m",  # Display formatted duration
        'journey_day': journey_day,
        'journey_month': journey_month
    })


def user_index(request):
    if request.method == 'POST':
        source = request.POST.get('source')
        destination = request.POST.get('destination')  # Changed 'to' to 'destination'
        airline = request.POST.get('airline')

        # Get dept_time and arr_time as strings
        dept_time_str = request.POST.get('dept_time')
        arr_time_str = request.POST.get('arr_time')
        
        dept_time = datetime.strptime(dept_time_str, '%Y-%m-%dT%H:%M')
        arr_time = datetime.strptime(arr_time_str, '%Y-%m-%dT%H:%M')

        # Create a Flight object
        obj = Flight.objects.create(
            source=source,
            destination=destination,
            airline=airline,
            dept_time=dept_time,
            stops=int(request.POST.get('stops')),  # Ensure stops are stored as an integer
            arr_time=arr_time
        )
        

        return redirect('Predict', flight_id=obj.id)

    return render(request, 'user/user-index.html')

def user_myprofile(request):
    user_id = request.session['user_id']
    user = UserModel.objects.get(user_id=user_id)

    if request.method == 'POST':
            username = request.POST.get("user_username")
            userppnum=request.POST.get('user_passportnumber')
            email = request.POST.get("user_email")
            contact = request.POST.get("user_contact")
            password = request.POST.get("user_password")
            address=request.POST.get('user_address')
            print(username,userppnum,email,contact,password,address)
            
            if len(request.FILES) != 0:
                
                        image = request.FILES["user_image"]
                        
                        user.user_passportnumber=userppnum
                        user.user_username = username
                        user.user_contact = contact
                        user.user_email=email
                        user.user_password = password
                        user.user_image = image
                        user.user_address=address
                        user.save()
                        messages.success(request,'Updated Successfully')
            else:
                        user.user_username = username
                        user.user_passportnumber=userppnum
                        user.user_contact = contact
                        user.user_contact = contact
                        user.user_email=email
                        # user.user_image=image
                        user.user_password = password
                        user.user_address=address
                        user.save()
                        messages.success(request,'Updated Successfully')
            
                        
            return redirect('user_myprofile')
    
    return render(request,'user/user-myprofile.html',{'user':user})
# userapp/views.py

# userapp/views.py

import numpy as np
import torch
from django.shortcuts import render
# from .models import Flight  # Ensure this model exists and is imported
from .gan import Generator, Discriminator  # Import your GAN models

