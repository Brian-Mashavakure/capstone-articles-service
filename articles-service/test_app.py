import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import patch, Mock
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from app import app

client = TestClient(app)

def test_get_articles_exception():
    with patch('app.getcancerarticles', side_effect=Exception("Error occurred")):
        response = client.get('/capstone/api/articles')
        assert response.status_code == 500
        assert response.json() == {"detail": "Error occurred"}


def test_predict_success():
    img = Image.new('RGB', (128, 128), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.8, 0.2]])

    with patch('app.model', mock_model):
        response = client.post('/capstone/api/model', data=img_byte_arr)
        assert response.status_code == 200
        assert response.json() == {"prediction": "malignant"}


def test_predict_invalid_image():
    invalid_data = b"this is not an image"
    response = client.post('/capstone/api/model', data=invalid_data)
    assert response.status_code == 500


def test_predict_model_exception():
    img = Image.new('RGB', (128, 128), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    with patch('app.model.predict', side_effect=Exception("Model error")):
        response = client.post('/capstone/api/model', data=img_byte_arr)
        assert response.status_code == 500
        assert response.json() == {"detail": "Model error"}
