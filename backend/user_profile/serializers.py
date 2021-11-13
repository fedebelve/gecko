from rest_framework import serializers
from user_profile.models import Profile
from django.contrib.auth.models import User

class UserSigninSerializer(serializers.Serializer):
    username = serializers.CharField(required = True)
    password = serializers.CharField(required = True)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        fields = ['nro_doc', 'country', 'birth_date', 'job_type', 'institution', 'user_plan']
        
        