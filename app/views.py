import json

from scrapping.logics import DataCollecter, StorageManager
from scrapping.topicmodeling import TopicModeler
from datetime import datetime

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
from django.shortcuts import render, redirect

from django.contrib.auth.models import User
from django.forms.utils import ErrorList
from .forms import PostForm

@login_required(login_url="/login/")
def index(request):
    
    context = {}
    context['segment'] = 'index'

    html_template = loader.get_template( 'index.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        
        load_template      = request.path.split('/')[-1]
        context['segment'] = load_template

        json_records = '[{"index":0,"Topic":1,"Number_of_Documents":3440,"Topic_Contribution":17.87,"Client_Satisfaction":38.0},{"index":1,"Topic":2,"Number_of_Documents":361,"Topic_Contribution":1.88,"Client_Satisfaction":47.0},{"index":2,"Topic":3,"Number_of_Documents":1304,"Topic_Contribution":6.77,"Client_Satisfaction":38.0},{"index":3,"Topic":4,"Number_of_Documents":533,"Topic_Contribution":2.77,"Client_Satisfaction":40.0},{"index":4,"Topic":5,"Number_of_Documents":495,"Topic_Contribution":2.57,"Client_Satisfaction":44.0},{"index":5,"Topic":6,"Number_of_Documents":645,"Topic_Contribution":3.35,"Client_Satisfaction":46.0},{"index":6,"Topic":7,"Number_of_Documents":312,"Topic_Contribution":1.62,"Client_Satisfaction":33.0},{"index":7,"Topic":8,"Number_of_Documents":1583,"Topic_Contribution":8.22,"Client_Satisfaction":51.0},{"index":8,"Topic":9,"Number_of_Documents":316,"Topic_Contribution":1.64,"Client_Satisfaction":36.0},{"index":9,"Topic":10,"Number_of_Documents":1586,"Topic_Contribution":8.24,"Client_Satisfaction":46.0},{"index":10,"Topic":11,"Number_of_Documents":541,"Topic_Contribution":2.81,"Client_Satisfaction":38.0},{"index":11,"Topic":12,"Number_of_Documents":730,"Topic_Contribution":3.79,"Client_Satisfaction":22.0},{"index":12,"Topic":13,"Number_of_Documents":823,"Topic_Contribution":4.28,"Client_Satisfaction":27.0},{"index":13,"Topic":14,"Number_of_Documents":385,"Topic_Contribution":2.0,"Client_Satisfaction":40.0},{"index":14,"Topic":15,"Number_of_Documents":270,"Topic_Contribution":1.4,"Client_Satisfaction":43.0},{"index":15,"Topic":16,"Number_of_Documents":1650,"Topic_Contribution":8.57,"Client_Satisfaction":47.0},{"index":16,"Topic":17,"Number_of_Documents":745,"Topic_Contribution":3.87,"Client_Satisfaction":42.0},{"index":17,"Topic":18,"Number_of_Documents":507,"Topic_Contribution":2.63,"Client_Satisfaction":50.0},{"index":18,"Topic":19,"Number_of_Documents":434,"Topic_Contribution":2.25,"Client_Satisfaction":36.0},{"index":19,"Topic":20,"Number_of_Documents":674,"Topic_Contribution":3.5,"Client_Satisfaction":43.0},{"index":20,"Topic":21,"Number_of_Documents":863,"Topic_Contribution":4.48,"Client_Satisfaction":45.0},{"index":21,"Topic":22,"Number_of_Documents":1052,"Topic_Contribution":5.47,"Client_Satisfaction":45.0}]'
        data = []
        data = json.loads(json_records)

        # if it is the dashboard then load the data
        if(load_template == 'dashbord.html'):
            context['taux_positif'] = 63
            context['taux_negatif'] = 37
            context['nombre_document'] = '112K'
            context['nombre_sujet'] = 27
            context['data'] = data

        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def create_post(request):

    context = {}
    context['segment'] = 'index'

    html_template = loader.get_template( 'index.html' )

    if request.method == "POST":
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            # post.author = request.user
            post.save()

            # collect data
            produit = form.cleaned_data['produit']
            start_date = form.cleaned_data['start_date']
            local_file_name = collect_data(produit, start_date)

            # get the topics 
            get_topics(local_file_name)

            # save the data in azure storage for later use
            #upload_data(local_file_name)
            return redirect('/dashbord.html')
    else:
        form = PostForm()
    return HttpResponse(render(request, 'index.html', {'form': form}))

def collect_data(search, since):

    lang = 'fr'
    since = since.strftime("%Y-%m-%d")
    dataCollecter = DataCollecter()
    return dataCollecter.collect_tweet(search, since, lang)

def upload_data(file_path):
    storageManager = StorageManager()
    storageManager.uploadData(file_path)

def get_topics(file_path):
    topicModeler = TopicModeler()
    topicModeler.get_topics(file_path)