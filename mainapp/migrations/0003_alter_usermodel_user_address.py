# Generated by Django 5.0.1 on 2024-10-02 03:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0002_alter_usermodel_user_address'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usermodel',
            name='user_address',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
