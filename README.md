# Animal Detector Flask API

This Flask server accepts a base64-encoded image and classifies it using a TensorFlow Lite model.

## POST /upload_interrupt

**Request:**
```json
{
  "image": "<base64_string>",
  "imagename": "capture1.jpg",
  "distance": 85
}
