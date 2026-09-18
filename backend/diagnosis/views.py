from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.inference import predict_from_bytes

from .models import PredictionRecord
from .serializers import PredictionRecordSerializer, PredictionUploadSerializer


class PredictView(APIView):
    """
    POST /api/predict/  (multipart/form-data, field name: "image")

    Runs the retinal image through the trained model, stores the result
    against the authenticated user, and returns the prediction.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        upload_serializer = PredictionUploadSerializer(data=request.data)
        upload_serializer.is_valid(raise_exception=True)
        image_file = upload_serializer.validated_data["image"]

        try:
            image_bytes = image_file.read()
            result = predict_from_bytes(image_bytes)
        except Exception as exc:  # noqa: BLE001 - surface a clean API error
            return Response({"error": f"Prediction failed: {exc}"}, status=status.HTTP_400_BAD_REQUEST)

        image_file.seek(0)  # rewind so it can be saved to the model field
        record = PredictionRecord.objects.create(
            user=request.user,
            image=image_file,
            predicted_class=result["predicted_class"],
            dr_probability=result["dr_probability"],
            no_dr_probability=result["no_dr_probability"],
        )

        serializer = PredictionRecordSerializer(record, context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class HistoryView(generics.ListAPIView):
    """GET /api/history/ -> the authenticated user's past predictions, newest first."""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = PredictionRecordSerializer

    def get_queryset(self):
        return PredictionRecord.objects.filter(user=self.request.user)
