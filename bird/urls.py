# bird/urls.py
from django.urls import path
from . import views

app_name = 'bird'
urlpatterns = [
    path('', views.index, name='index'),         # if your base points to bird:index
    path('home/', views.home, name='home'),      # gallery / homepage with 20 images
    path('image/', views.image_upload, name='image_upload'),  # upload form
    # path('result/', views.result, name='result'),            # redirect or result page (optional)
]








# # bird/urls.py
# from django.urls import path
# from . import views

# app_name = "bird" 

# urlpatterns = [
#     path('',views.index,name='index'),
#     path("home/", views.home, name="home"),
#     path("image/", views.image_upload, name="image_upload"),
#     path("audio/", views.audio_upload, name="audio_upload"),
#     # Optional APIs for JS frontends
#     path("api/predict/image", views.api_predict_image, name="api_predict_image"),
#     path("api/predict/audio", views.api_predict_audio, name="api_predict_audio"),
# ]
