from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import SignupView, MeView
from .token_views import EmailOrUsernameTokenObtainPairView

urlpatterns = [
    path("signup/", SignupView.as_view(), name="signup"),
    path("login/", EmailOrUsernameTokenObtainPairView.as_view(), name="login"),
    path("refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("me/", MeView.as_view(), name="me"),
]
