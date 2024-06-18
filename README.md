# Introduction
The API provides two endpoints for processing PDF documents related to glass patents. These endpoints enable users to extract metadata and glass configurations from PDF files, with the option for additional Optical Character Recognition (OCR) processing.

Additionally, a rudimentary user interface is provided where PDF files can be uploaded to one of the endpoints. The extraction result is shown in a table and can be downloaded as a CSV and HTML file.

# API (server-glass/app.py)

## /show
- **Method:** GET
- **Description:** Provides the user interface, see section "User Interface"

## /update_knowledge
- **Method:** POST
- **Description:**: Updates the knowledge base
- **Request Parameters:**
  - `json`: raw json data in format {"replacement": ["regex1", "string1", ...]}
  - OR: `file`: JSON file containing the replacement data
- **Response:**
  - **Status Code:** 200 OK
- **Error Responses:**
  - **Status Code:** 400 Bad Request
    - **Description:** No valid json data found in the request.
  - **Status Code:** 500 Internal Server Error
    - **Description:** Error updating the knowledge base.

## /single_pdf
- **Method:** POST
- **Description:** Extracts metadata and glass configurations from a single PDF file.
- **Request Parameters:**
  - `file`: PDF file containing glass patent information (sent as a multipart/form-data)
- **Response:**
  - **Status Code:** 200 OK
  - **Content:** JSON dictionary containing metadata and glass configurations.
- **Error Responses:**
  - **Status Code:** 400 Bad Request
    - **Description:** No file found in the request.
  - **Status Code:** 500 Internal Server Error
    - **Description:** Error processing the document.

## /single_pdf_ocr
- **Method:** POST
- **Description:** Same as /single_pdf, with additional OCR processing for improved text extraction.
- **Request Parameters:**
  - `file`: PDF file containing glass patent information (sent as a multipart/form-data)
- **Response:**
  - **Status Code:** 200 OK
  - **Content:** JSON dictionary containing metadata, glass configurations, and OCR-enhanced text (if applicable).
- **Error Responses:**
  - **Status Code:** 400 Bad Request
    - **Description:** No file found in the request.
  - **Status Code:** 500 Internal Server Error
    - **Description:** Error processing the document.
- **Note:** OCR has a high computational cost and is therefore slow. The current implementation is cpu-only.

## /single_pdf_async
- **Method:** POST
- **Description:** Schedules a task to extract metadata and glass configurations from a single PDF file asynchronously.
- **Request Parameters:**
  - `file`: PDF file containing glass patent information (sent as a multipart/form-data)
- **Response:**
  - **Status Code:** 202 Accepted
  - **Content:** JSON dictionary containing task ID or error information.
- **Error Responses:**
  - **Status Code:** 400 Bad Request
    - **Description:** No file found in the request.
  - **Status Code:** 500 Internal Server Error
    - **Description:** Error scheduling the task.

## /single_pdf_ocr_async
- **Method:** POST
- **Description:** Schedules a task to extract metadata and glass configurations with OCR processing from a single PDF file asynchronously.
- **Request Parameters:**
  - `file`: PDF file containing glass patent information (sent as a multipart/form-data)
- **Response:**
  - **Status Code:** 202 Accepted
  - **Content:** JSON dictionary containing task ID or error information.
- **Error Responses:**
  - **Status Code:** 400 Bad Request
    - **Description:** No file found in the request.
  - **Status Code:** 500 Internal Server Error
    - **Description:** Error scheduling the task.

## /status/<task_id>
- **Method:** GET
- **Description:** Queries the current status of a scheduled task.
- **Request Parameters:** None
- **Response:**
  - **Status Code:** 200 OK
  - **Content:** JSON dictionary containing the task status and result if the task is done.
  - **Possible States:** `done`, `failed`
  - **Result (if done):** Same as `/single_pdf`
- **Error Responses:**
  - **Status Code:** 404 Not Found
    - **Description:** Task ID not found.
  - **Status Code:** 500 Internal Server Error
    - **Description:** Error querying the task status.
