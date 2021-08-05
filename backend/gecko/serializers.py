from rest_framework import serializers

class AnalizeSerializer(serializers.Serializer):
    InMemoryUploadedFile = serializers.ImageField()