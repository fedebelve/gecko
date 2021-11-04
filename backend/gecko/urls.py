"""gecko URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from gecko import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls')),
    path('signup', views.signup),
    path('signin', views.signin),
    path('analize/', views.AnalizeBase64.as_view()),
    #path('check/', views.CheckImage.as_view()),
    path('validate-image/', views.CheckValidImage.as_view()),
    path('validate-quality/', views.CheckImageQuality.as_view()),
    path('preprocess/', views.PreprocessImage.as_view()),
    path('bentransformation/', views.BenTransformation.as_view()),
    path('process/', views.ProcessImage.as_view()),
]