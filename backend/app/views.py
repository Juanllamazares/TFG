from django.shortcuts import render


# Create your views here.
def index(request):
    context_dict = {}
    return render(request, 'pages/index.html', context=context_dict)


def dashboard(request):
    context_dict = {}
    return render(request, 'pages/dashboard.html', context=context_dict)
