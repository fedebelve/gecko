from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    nro_doc = models.PositiveIntegerField(null=False)
    country = models.CharField(max_length=50, null=False)
    birth_date = models.DateField(null=False)
    job_type = models.CharField(max_length=20,null=False,choices=[('PA', 'Particular'), ('PU', 'Público'), ('PR', 'Privado')])
    institution = models.CharField(max_length=150,null=True)
    user_plan = models.CharField(max_length=20,null=False,choices=[('FR', 'Free'), ('BA', 'Básico'), ('PR', 'Premium')], default='FR')

class ConsumedService(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    analized_images = models.PositiveIntegerField(null=False, default=0)
    _last_modified = models.DateTimeField(auto_now=True)