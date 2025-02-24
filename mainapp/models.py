from django.db import models

# Create your models here.
class UserModel(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_username = models.CharField(max_length=100)
    user_passportnumber=models.CharField(max_length=20)
    user_email = models.EmailField(max_length=100)
    user_password = models.CharField(max_length=100)
    user_contact = models.CharField(max_length=15)
    user_address = models.CharField(max_length=255, null=True, blank=True)
    user_image = models.ImageField(upload_to='user/images')
 
    class Meta:
        db_table = 'User_Details'
    

class PredModel(models.Model):
    id = models.AutoField(primary_key=True)
    source = models.CharField(max_length=100)
    to = models.CharField(max_length=100)
    airline= models.CharField(max_length=100)
    dept_time = models.DateTimeField(max_length=50)
    stops=models.IntegerField()
    arr_time=models.DateTimeField(max_length=50)

    class Meta:
        db_table = 'predmodel'

class TestingModel(models.Model):
    id = models.AutoField(primary_key=True)
    Total_Stops=models.IntegerField()
    Air_India=models.IntegerField()
    GoAir=models.IntegerField()
    IndiGo=models.IntegerField()
    Jet_Airways=models.IntegerField()
    Jet_Airways_Business=models.IntegerField()
    Multiple_carriers=models.IntegerField()
    Multiple_carriers_Premium_economy=models.IntegerField()
    SpiceJet=models.IntegerField()
    Trujet=models.IntegerField()
    Vistara=models.IntegerField()
    Vistara_Premium_economy=models.IntegerField()
    Chennai=models.IntegerField()
    Delhi=models.IntegerField()
    Kolkata=models.IntegerField()
    Mumbai=models.IntegerField()
    Cochin=models.IntegerField()
    Hyderabad=models.IntegerField()
    journey_day=models.IntegerField()
    journey_month=models.IntegerField()
    Dep_Time_hour=models.IntegerField()
    Dep_Time_min=models.IntegerField()
    Arrival_Time_hour=models.IntegerField()
    Arrival_Time_min=models.IntegerField()
    dur_hour=models.IntegerField()
    dur_min=models.IntegerField()


    class Meta:
        db_table='datatestingmodel'

# userapp/models.py



class Flight(models.Model):
    source = models.CharField(max_length=100)
    destination = models.CharField(max_length=100, null=False)  # Changed 'to' to 'destination' for clarity
    airline = models.CharField(max_length=100)
    dept_time = models.DateTimeField() # Assuming this is a time field
    stops = models.IntegerField()  # Assuming this represents the number of stops
    arr_time = models.DateTimeField()  # Assuming this is also a time field

    def __str__(self):
        return f"{self.airline} flight from {self.source} to {self.destination}"

    Air_India = models.BooleanField(default=False)
    GoAir = models.BooleanField(default=False)
    IndiGo = models.BooleanField(default=False)
    Jet_Airways = models.BooleanField(default=False)
    Jet_Airways_Business = models.BooleanField(default=False)
    Multiple_carriers = models.BooleanField(default=False)
    Multiple_carriers_Premium_economy = models.BooleanField(default=False)
    SpiceJet = models.BooleanField(default=False)
    Trujet = models.BooleanField(default=False)

    # Example of one-hot encoded fields for cities
    Bangalore = models.BooleanField(default=False)
    Hyderabad = models.BooleanField(default=False)
    Kolkata = models.BooleanField(default=False)
    Delhi = models.BooleanField(default=False)
    Cochin = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.airline} flight from {self.source} to {self.destination} departing at {self.dept_time}"

















 