from rest_framework import serializers

from .models import PredictionRecord


class PredictionRecordSerializer(serializers.ModelSerializer):
    confidence = serializers.FloatField(read_only=True)

    class Meta:
        model = PredictionRecord
        fields = (
            "id",
            "image",
            "predicted_class",
            "dr_probability",
            "no_dr_probability",
            "confidence",
            "created_at",
        )
        read_only_fields = fields


class PredictionUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
