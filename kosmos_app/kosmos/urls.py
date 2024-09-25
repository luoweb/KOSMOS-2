from django.urls import path
from .views import index, kosmos_api

urlpatterns = [
    path("", index, name="index"),
    path("api", kosmos_api, name="kosmos_api"),
]
