from django.conf import settings
from django.db import models


def prediction_image_path(instance, filename):
    return f"predictions/user_{instance.user_id}/{filename}"


class PredictionRecord(models.Model):
    """One retinal image prediction, tied to the user who uploaded it."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="predictions"
    )
    image = models.ImageField(upload_to=prediction_image_path)
    predicted_class = models.CharField(max_length=32)
    dr_probability = models.FloatField()
    no_dr_probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} - {self.predicted_class} ({self.created_at:%Y-%m-%d %H:%M})"

    @property
    def confidence(self):
        return max(self.dr_probability, self.no_dr_probability)
