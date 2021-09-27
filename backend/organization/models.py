from django.db import models
from rest_framework_api_key.models import AbstractAPIKey
# Create your models here.

class Organization(models.Model):
    name = models.CharField(max_length=128)
    description = models.TextField(null=True)
    active = models.BooleanField(default=True)

class OrganizationAPIKey(AbstractAPIKey):
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="api_keys")
    