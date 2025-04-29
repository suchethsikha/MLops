FROM python:3.9-slim

WORKDIR /app

# Copy app code and model
COPY ./RedditApp.py .
COPY reddit_model_pipeline.joblib .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]