# bird/forms.py
from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(required=True)

class AudioUploadForm(forms.Form):
    audio = forms.FileField(required=True)
