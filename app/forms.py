from django.forms import ModelForm
from .models import Post 
from django import forms

# Create the form class.
class PostForm(ModelForm):

    produit = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder" : "e.g : Redmi Note 8",                
                "class": "form-control"
            }
        ))
    start_date = forms.DateField(
        widget=forms.DateInput(
            attrs={
                'type': 'date',
                'class': 'form-control'}))
    end_date = forms.DateField(
        widget=forms.DateInput(
            attrs={
                'type': 'date',
                'class': 'form-control'}))
    termes = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'placeholder' : "Les aspects qui vous intéressent le plus à propos du produis",
                "class": "form-control",
                'rows':4}),
        initial = "batterie, prix, photos, qualité, support, chargeur, mémoire")

    class Meta:
        model = Post
        fields = ['produit', 'start_date', 'end_date', 'termes']