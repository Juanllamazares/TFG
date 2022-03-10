from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt



# Create your views here.
def index(request):
    context_dict = {}
    return render(request, 'pages/index.html', context=context_dict)


@csrf_exempt
def dashboard(request):
    context_dict = {}
    if request.method == 'POST':
        context_dict["show_rmse_calculation"] = True
        context_dict["show_mape_calculation"] = True
        context_dict["rmse_value"] = 10.2
        context_dict["mape_value"] = 3

        print("example")
    return render(request, 'pages/dashboard.html', context=context_dict)
