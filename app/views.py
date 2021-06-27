from scrapping.logics import DataCollecter
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
            collect_data(produit, start_date)
            return redirect('/dashbord.html')
    else:
        form = PostForm()
    return HttpResponse(render(request, 'index.html', {'form': form}))

def collect_data(search, since):

    lang = 'fr'
    since = since.strftime("%Y-%m-%d")
    dataCollecter = DataCollecter()
    local_file_name = dataCollecter.collect_tweet(search, since, lang)