from django.contrib import admin
from django.urls import path
from explainable_ai import views as mainView
from admins import views as admins
from users import views as usr
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib.staticfiles.urls import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('UserLogout/', usr.UserLogout, name='UserLogout'),
    path('auto_prediction/', usr.auto_prediction, name='auto_prediction'),
    path('network_simulation/', usr.network_simulation, name='network_simulation'),
    path('', mainView.index, name='index'),
    path('index', mainView.index, name='index_alt'),
    path('AdminLogincheck', admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('UserLogin', mainView.UserLogin, name='UserLogin'),

    # Admin views
    path('userDetails', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivUsers/', admins.ActivaUsers, name='activate_users'),
    path('DeleteUsers/', admins.DeleteUsers, name='delete_users'),
    path('adminhome/', admins.adminhome, name='adminhome'),

    # User views
    path('UserRegisterForm', usr.UserRegisterActions, name='UserRegisterForm'),
    path('UserLoginCheck/', usr.UserLoginCheck, name='UserLoginCheck'),
    path('UserHome/', usr.UserHome, name='UserHome'),
    path('ViewDataset/', usr.ViewDataset, name='ViewDataset'),
    path('prediction/', usr.prediction, name='prediction'),
    path('training', usr.training, name='training'),
    path('upload_data/', usr.upload_data_view, name='upload_data'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
