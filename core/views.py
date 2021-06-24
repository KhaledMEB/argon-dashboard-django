from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.forms.utils import ErrorList
from django.http import HttpResponse
from .forms import CollectDataForm

def collect_data_view(request):
    form = CollectDataForm()
     return render(request, "/index.html", {"form": form})
    # if request.method == "POST":

    #     if form.is_valid():
    #         search_txt = form.cleaned_data.get("search_txt")
    #         date_since = form.cleaned_data.get("date_since")   

    # return render(request, "/index.html", {"form": form})```