"""customer_segmentation_new URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
import os
from softcom.views import *

urlpatterns = [
    path('admin/', admin.site.urls,name='/'),
    path('',preprocessing, name='preprocessing'),
    path('preprocessing/', preprocessing, name='preprocessing'),
    path('checker_page/', checker_page, name='checker_page'),
    # path('elbow_graph/', elbow_graph, name='elbow_graph'),
    path('clustering/', clustering, name='clustering'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
