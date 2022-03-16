from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
def home(request):
    context_dict = {}
    return render(request, 'home.html', context=context_dict)


@csrf_exempt
def dashboard(request):
    context_dict = {}
    if request.method == 'POST':
        context_dict["show_rmse_calculation"] = True
        context_dict["show_mape_calculation"] = True
        context_dict["rmse_value"] = 10.2
        context_dict["mape_value"] = 3
    else:
        context_dict["test"] = 0
    return render(request, 'dashboard.html', context=context_dict)


def login(request):
    context_dict = {}
    return render(request, 'login.html', context=context_dict)


def sign_up(request):
    context_dict = {}
    return render(request, 'signup.html', context=context_dict)


def profile(request):
    context_dict = {}
    return render(request, 'profile.html', context=context_dict)
