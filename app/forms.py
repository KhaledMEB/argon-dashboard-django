from django.forms import ModelForm
from .models import Post 
from django import forms

# Create the form class.
class PostForm(ModelForm):
    class Meta:
        model = Post
        fields = ['produit', 'start_date', 'end_date', 'termes']