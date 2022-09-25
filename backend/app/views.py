import json

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import datetime
import main
import auth


# Create your views here.
def home(request):
    context_dict = {}
    return render(request, 'home.html', context=context_dict)


@csrf_exempt
def dashboard(request):
    context_dict = {}
    if request.method == 'POST':
        symbol = request.POST.get("symbol")
        n_days = 365
        data = main.get_stock_price(symbol)

        date_list = list(data.keys())
        close_price_list = list(data.values())
        date_list.reverse()
        close_price_list.reverse()
        close_price_list = close_price_list[-n_days:]
        date_list = date_list[-n_days:]
        results = main.stock_prediction_lstm(symbol, n_days)

        stock_dict = {
            "symbol": symbol,
            "labels": json.dumps(date_list),
            "data": json.dumps(close_price_list),
            "predicted_data": json.dumps(results["predicted_data"])
        }

        first_date = datetime.datetime.strptime(date_list[0], '%Y-%m-%d')
        last_date = datetime.datetime.strptime(date_list[-1], '%Y-%m-%d')

        context_dict = {
            "full_view": True,
            "rmse_value": 10.2,
            "mape_value": round(results["mape_train"], 2),
            "stock": stock_dict,
            "labels": json.dumps(date_list),
            "first_date": first_date.strftime("%m/%Y"),
            "last_date": last_date.strftime("%m/%Y"),
        }
    else:
        context_dict["test"] = 0
    return render(request, 'dashboard.html', context=context_dict)


def login(request):
    context_dict = {}
    if request.method == 'POST':
        email = request.POST.get("email")
        password = request.POST.get("password")
        auth.login(email, password)

    return render(request, 'login.html', context=context_dict)


def sign_up(request):
    context_dict = {}
    if request.method == 'POST':
        email = request.POST.get("email")
        password = request.POST.get("password")
        auth.sign_up(email, password)

    return render(request, 'signup.html', context=context_dict)


def profile(request):
    context_dict = {}
    return render(request, 'profile.html', context=context_dict)
