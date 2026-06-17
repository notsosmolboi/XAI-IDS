from django.db import models

class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    mobile = models.CharField(max_length=15)
    password = models.CharField(max_length=100)
    status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'user_registration'


class DetectionLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField()
    is_attack = models.BooleanField(default=False)
    # Store the core features for XAI auditing later
    flow_duration = models.FloatField()
    packet_rate = models.FloatField()
    
    def __str__(self):
        return f"{self.prediction} - {self.timestamp}"

    class Meta:
        db_table = 'detection_logs'