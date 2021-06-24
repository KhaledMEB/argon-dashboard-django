# -*- encoding: utf-8 -*-

from django.urls import path, re_path
from app import views

urlpatterns = [

    # The home page
    path('', views.create_post, name='home'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
