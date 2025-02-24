from django.db import models

# Create your models here.
class Dataset(models.Model):
    data_id = models.AutoField(primary_key=True)
    data_set = models.FileField(upload_to='files/')
    lr_Accuracy = models.FloatField(null=True)
    lr_algo = models.CharField(max_length=50,null=True)
    knn_Accuracy = models.FloatField(null=True)
    knn_algo = models.CharField(max_length=50,null=True)
    svr_Accuracy = models.FloatField(null=True)
    svr_algo = models.CharField(max_length=50,null=True)
    rf_Accuracy = models.FloatField(null=True)
    rf_algo = models.CharField(max_length=50,null=True)
    dt_Accuracy = models.FloatField(null=True)
    dt_algo = models.CharField(max_length=50,null=True)
    class Meta:
        db_table = 'dataset'


from django.db import models

class GAN(models.Model):
    Generated_Data = models.FloatField(default=0.0)
    Name = models.CharField(max_length=255, default="Unnamed GAN")  # Set a default value

    def __str__(self):
        return self.Name
22





class RNN(models.Model):
    Mean_Squared_Error = models.FloatField()
    Mean_Absolute_Error = models.FloatField()
    R2_Score = models.FloatField()
    Name = models.CharField(max_length=100)
    # Add other fields as necessary

    def __str__(self):
        return self.Name


