from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _
import datetime #for checking renewal date range.

class CollectDataForm(forms.Form):
    search_txt = forms.CharField(
        help_text="Entrez le nom du produit cible",
        required=True)

    date_since = forms.DateField(
        help_text="Collecter des données depuis")

    def clean_date_since(self):
        data = self.cleaned_data['date_since']

        #Check date is not in futur.
        if data > datetime.date.today():
            raise ValidationError(_('Date non valide'))

        #Check date is in range of 9 months in the past.
        if data < datetime.date.today() - datetime.timedelta(weeks=40):
            raise ValidationError(_('Date non valide - Vous ne pouvez pas collecter plus de 9 mois'))

        # Remember to always return the cleaned data.
        return data
