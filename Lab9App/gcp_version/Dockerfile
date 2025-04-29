FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
# Copy app code and model


# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]