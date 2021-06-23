# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.forms.utils import ErrorList
from django.http import HttpResponse
from .forms import CollectDataForm

def collect_data_view(request):
    form = CollectDataForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            search_txt = form.cleaned_data.get("search_txt")
            date_since = form.cleaned_data.get("date_since")
            if user is not None:
                login(request, user)
                return redirect("/")
            else:    
                msg = search_txt + ' ' + date_since    
        else:
            msg = 'Erreur de validation de la requête'    

    return render(request, "/index.html", {"form": form, "msg" : msg})