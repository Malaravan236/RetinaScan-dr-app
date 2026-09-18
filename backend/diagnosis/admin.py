from django.contrib import admin

from .models import PredictionRecord


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "predicted_class", "dr_probability", "no_dr_probability", "created_at")
    list_filter = ("predicted_class",)
    search_fields = ("user__username",)
