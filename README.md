
# ICD Code Extraction API - README

## Overview

This project provides a FastAPI-based web service for extracting ICD (International Classification of Diseases) codes. It supports extraction workflows using either OCR-generated page data or encounter IDs. The application includes endpoints for integration with external systems via Dapr input bindings, enabling seamless microservice communication. It also features utilities for health checks and pod lifecycle management.

---

## Project Structure

```
.
├── app.py               # Main application entry point with API routes
├── icd_model.py         # Core logic for ICD code extraction and workflow handling
├── utils/
│   └── dapr_utils.py    # Utility for Dapr state management
├── constants.py         # Constants for health state conditions
└── README.md            # Project documentation (this file)
```

---

## Features

1. **ICD Code Extraction**:
   - **Using OCR Pages**: Extracts ICD codes directly from OCR-processed medical documents.
   - **Using Encounter IDs**: Fetches and processes ICD codes based on medical record and encounter IDs.

2. **Dapr Integration**:
   - Exposes endpoints for Dapr input bindings to enable inter-service communication.

3. **Health Management**:
   - `healthz` endpoint for health checks.
   - `prestop` hook to handle graceful pod termination.

4. **Multithreading**:
   - Initializes background tasks for testing and loading workflows.

---

## API Endpoints

### 1. `/ml-icd-extraction` (POST)
Extracts ICD codes using encounter IDs.

- **Payload**:
  ```json
  {
    "encounterIds": ["E123", "E456"],
    "medicalRecordId": "MR001",
    "batchYear": 2023,
    "plan": "Standard"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "data": <extracted_icd_codes>
  }
  ```

### 2. `/healthz` (GET)
Returns a 200 HTTP status to indicate service health.

- **Response**:
  ```json
  {
    "success": true
  }
  ```

### 3. `/prestop` (GET)
Handles pod termination and updates service health to a degraded state.

- **Response**:
  ```json
  {
    "success": true
  }
  ```

### 4. `/ml-icd-extraction` (OPTIONS)
Dapr input binding endpoint for the ML-based ICD extraction pipeline.

- **Response**:
  ```json
  {
    "success": true
  }
  ```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Required libraries (listed in `requirements.txt`):
  - FastAPI
  - Pydantic
  - Additional dependencies from `icd_model.py`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. Access the API:
   - Visit `http://localhost:8000/docs` for the Swagger UI documentation.

---

## Usage

### 1. Test ICD Extraction
Run the application and send POST requests to the `/ml-icd-extraction` endpoint with appropriate payloads.

### 2. Health Check
Use the `/healthz` endpoint to verify the service is running correctly.

### 3. Debugging
Logs are printed to the console for monitoring API requests and background tasks.

---

## Future Improvements
1. Add detailed error handling and logging.
2. Include unit tests for `icd_model.py` logic.
3. Extend support for additional ICD code workflows.
4. Optimize threading for better scalability.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.