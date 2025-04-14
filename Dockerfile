# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure smolagents is installed properly
RUN pip install --no-cache-dir smolagents

# Copy the current directory contents into the container at /app
COPY . .

# Make port 7860 available to the world outside this container (for Gradio UI)
EXPOSE 7860

# Define environment variables (placeholders, should be set in docker-compose or during 'docker run')
ENV GEMINI_API_KEY=your_gemini_api_key_here
ENV OPENAI_API_KEY=your_openai_api_key_here

# Run main.py when the container launches
CMD ["python", "-u", "./main.py"]
