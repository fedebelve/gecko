from django.contrib import admin
from rest_framework_api_key.admin import APIKeyModelAdmin
from .models import Organization, OrganizationAPIKey
# Register your models here.
@admin.register(OrganizationAPIKey)
class OrganizationAPIKeyModelAdmin(APIKeyModelAdmin):
    pass

@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    pass