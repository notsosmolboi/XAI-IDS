from django.shortcuts import render, redirect
from django.contrib import messages
from users.models import UserRegistrationModel


def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        if usrid == 'admin' and pswd == 'admin':
            total_users = UserRegistrationModel.objects.count()
            activated_users = UserRegistrationModel.objects.filter(status='activated').count()
            pending_users = UserRegistrationModel.objects.filter(status='pending').count()
            context = {
                'total_users': total_users,
                'activated_users': activated_users,
                'pending_users': pending_users,
            }
            return render(request, 'admins/AdminHome.html', context)
        else:
            messages.error(request, 'Invalid admin credentials. Please check your login details.')
    return render(request, 'AdminLogin.html', {})


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})


def ActivaUsers(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        if user_id:
            UserRegistrationModel.objects.filter(id=user_id).update(status='activated')
    return redirect('RegisterUsersView')


def DeleteUsers(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        if user_id:
            UserRegistrationModel.objects.filter(id=user_id).delete()
    return redirect('RegisterUsersView')


def adminhome(request):
    total_users = UserRegistrationModel.objects.count()
    activated_users = UserRegistrationModel.objects.filter(status='activated').count()
    pending_users = UserRegistrationModel.objects.filter(status='pending').count()
    
    context = {
        'total_users': total_users,
        'activated_users': activated_users,
        'pending_users': pending_users,
    }
    return render(request, 'admins/AdminHome.html', context)