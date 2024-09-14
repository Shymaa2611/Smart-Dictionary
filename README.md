# Smart Dictionary

**Smart Dictionary** is a web application designed to provide comprehensive word details including definitions, examples, antonyms, translations to Arabic, and word types (e.g., noun, verb, adverb). The application utilizes FastAPI for the backend and Jinja for templating, and integrates fine-tuned BERT and Gamma models for its functionalities.

## Features

- **Word Definition:** Get the definition of a word.
- **Example:** Retrieve examples of how the word is used in context.
- **Antonym:** Get antonyms of the word.
- **Arabic Translation:** Translate the word into Arabic.
- **Word Type:** Determine the part of speech (e.g., noun, verb, adverb) using a fine-tuned BERT model.

## Technologies

- **FastAPI:** For building the web API.
- **Jinja2:** For HTML templating.
- **BERT:** Fine-tuned for determining the type of word.
- **Gamma:** Fine-tuned for retrieving antonyms.

## Installation

### Prerequisites

Make sure you have Python 3.7 or higher installed. Also, ensure that you have `pip` installed for package management.

### Clone the Repository

```bash
git clone https://github.com/yourusername/smart-dictionary.git
cd smart-dictionary
```
Install Dependencies
- install the required packages:

```bash
pip install -r requirements.txt

```
Set Up Static and Template Directories

Ensure that you have the following directories in your project:

    static/ for static files like CSS and images.
    templates/ for HTML files.

## Download Models
You need to download and place the fine-tuned BERT and Gamma models in the appropriate   
directories.
- Bert Model That Fine-tuned for POS  can download Model Checkpoint from :https://drive.google.com/file/d/1NQ5Y4gUV2dLnmVH4QznQy2LTrC8JNDvd/view?usp=drive_link 
- Gamma Model That Fine-tuned for Antonym can download Model Checkpoint from :https://drive.google.com/file/d/1NThgMZQLc4hWvLbzzki7IO3W9rPY3eYF/view?usp=drive_link
- NLLB model That is used for egyptian translation can download Model Checkpoint from :https://drive.google.com/drive/folders/1VcZ4ZFHgIFyX6QuBmzPq6_x4y8QoVkp9?usp=drive_link

 
## Running the Application
To run the FastAPI application, use:
```bash
  uvicorn main:app --reload
```
By default, the application will be accessible at http://127.0.0.1:8000

## API Endpoints
 Swagger documents will be accessible at  http://127.0.0.1:8000/docs


## Templates and Static Files

    HTML Templates: Located in the templates/ directory.
    CSS and Images: Located in the static/ directory.

