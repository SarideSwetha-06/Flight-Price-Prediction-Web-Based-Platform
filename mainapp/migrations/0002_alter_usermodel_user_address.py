# Generated by Django 5.0.1 on 2024-10-02 03:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usermodel',
            name='user_address',
            field=models.TextField(max_length=50),
        ),
    ]
