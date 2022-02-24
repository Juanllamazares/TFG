from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=30)
    email = models.CharField(max_length=100)
    create_date = models.DateField()
